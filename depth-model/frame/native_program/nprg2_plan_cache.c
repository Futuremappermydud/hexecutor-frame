#include "nprg2_plan_cache.h"
#include <string.h>

static int handler_index_fast(const Nprg2Executor *e, uint8_t op, uint8_t variant) {
    uint32_t at = ((uint32_t)op * 37u + (uint32_t)variant * 17u) % NPRG2_EXEC_REGISTRY_CAPACITY;
    for (uint32_t n = 0; n < NPRG2_EXEC_REGISTRY_CAPACITY; n++, at = (at + 1u) % NPRG2_EXEC_REGISTRY_CAPACITY) {
        const Nprg2ExecHandler *h = &e->handlers[at];
        if (!h->used) return -1;
        if (h->opcode == op && h->variant == variant) return (int)at;
    }
    return -1;
}

size_t nprg2_plan_required_bytes(uint32_t command_count) {
    uint32_t cap = (command_count + 127u) & ~127u;
    return (size_t)NPRG2_PLAN_HEADER_BYTES + cap;
}

int nprg2_plan_build(const Nprg2Bound *bound,
                     const Nprg2PlanBuildConfig *config,
                     void *cache_arena,
                     size_t capacity,
                     Nprg2PlanCache *out) {
    if (!bound || !bound->ready || !config || !config->program || !config->executor ||
        !config->pools || !cache_arena || !out) {
        return NPRG2_PLAN_ARGUMENT;
    }
    memset(out, 0, sizeof *out);
    uint32_t cmd_count = bound->command.bytes / NPRG2_COMMAND_BYTES;
    if (!cmd_count || cmd_count > NPRG2_PLAN_MAX_COMMANDS) return NPRG2_PLAN_ARGUMENT;

    size_t needed = nprg2_plan_required_bytes(cmd_count);
    if (capacity < needed) return NPRG2_PLAN_CAPACITY;
    if ((uintptr_t)cache_arena % 8u) return NPRG2_PLAN_ARGUMENT;

    Nprg2PlanHeader *hdr = (Nprg2PlanHeader *)cache_arena;
    uint8_t *tags = (uint8_t *)cache_arena + NPRG2_PLAN_HEADER_BYTES;
    memset(cache_arena, 0, needed);
    memset(tags, NPRG2_PLAN_TAG_UNCOVERED, needed - NPRG2_PLAN_HEADER_BYTES);

    uint32_t max_scratch = 0;
    uint16_t dynamic_count = 0;

    /* Statically prove every command in the prepared schedule */
    for (uint32_t ix = 0; ix < cmd_count; ix++) {
        Nprg2Command cmd;
        if (nprg2_command_at(bound, ix, &cmd) != NPRG2_OK) return NPRG2_PLAN_REJECTED;

        int hi = handler_index_fast(config->executor, cmd.opcode, cmd.variant);
        if (hi < 0 || hi >= (int)NPRG2_EXEC_REGISTRY_CAPACITY) return NPRG2_PLAN_UNSUPPORTED;

        const Nprg2ExecHandler *h = &config->executor->handlers[hi];
        uint64_t req_scratch = (uint64_t)h->scratch_bytes + h->concurrent_worker_scratch_bytes;
        if (req_scratch > config->scratch_bytes) return NPRG2_PLAN_SCRATCH;
        if (req_scratch > max_scratch) max_scratch = (uint32_t)req_scratch;

        Nprg2ExecCall call = {
            .command_index = ix,
            .command = cmd,
            .program = config->program,
            .bound = bound,
            .pools = config->pools,
            .pool1_writable_bytes = config->pool1_writable_bytes,
            .inputs = config->inputs,
            .input_count = config->input_count,
            .outputs = config->outputs,
            .output_count = config->output_count,
            .scratch = config->scratch,
            .scratch_bytes = config->scratch_bytes,
            .scratch_region = {{(uint8_t *)config->scratch, config->scratch_bytes}}
        };
        /* The plan carries one scratch region; a handler registered for
         * another region fails closed here instead of getting the wrong one. */
        if (h->scratch_region != NPRG2_EXEC_SCRATCH_DEFAULT) return NPRG2_PLAN_UNSUPPORTED;

        /* Run full static validation (including alias_proof with caller scratch) */
        if (!h->validate(&call, h->ctx)) return NPRG2_PLAN_REJECTED;

        tags[ix] = (uint8_t)hi;
    }

    /* Populate serialized header */
    hdr->magic = NPRG2_PLAN_MAGIC;
    hdr->version = NPRG2_PLAN_VERSION;
    hdr->header_bytes = NPRG2_PLAN_HEADER_BYTES;
    hdr->total_bytes = (uint32_t)needed;
    hdr->command_count = cmd_count;
    hdr->command_capacity = (cmd_count + 127u) & ~127u;
    hdr->registry_count = config->executor->count;
    hdr->pool1_writable_bytes = config->pool1_writable_bytes;
    hdr->max_scratch_bytes = max_scratch;
    hdr->dynamic_command_count = dynamic_count;

    if (config->session_fingerprint) {
        memcpy(hdr->fingerprint, config->session_fingerprint, 32);
    } else {
        nprg2_sha256(bound->command.data, bound->command.bytes, hdr->fingerprint);
    }

    for (unsigned i = 0; i < 7; i++) {
        hdr->pool_bases[i] = (uint32_t)(uintptr_t)config->pools[i].base;
    }

    /* Populate runtime descriptor */
    out->header = hdr;
    out->tags = tags;
    out->command_count = cmd_count;
    out->pool1_writable_bytes = config->pool1_writable_bytes;
    memcpy(out->pools, config->pools, sizeof out->pools);
    out->arena = cache_arena;
    out->arena_capacity = capacity;
    out->ready = 1;

    return NPRG2_PLAN_OK;
}

int nprg2_plan_build_from_request(const Nprg2Executor *executor,
                                  const Nprg2ExecRequest *request,
                                  void *cache_arena,
                                  size_t capacity,
                                  Nprg2PlanCache *out) {
    if (!executor || !request) return NPRG2_PLAN_ARGUMENT;
    Nprg2PlanBuildConfig cfg = {
        .program = request->program,
        .executor = executor,
        .pools = request->pools,
        .pool1_writable_bytes = request->pool1_writable_bytes,
        .scratch = request->scratch,
        .scratch_bytes = request->scratch_bytes,
        .session_fingerprint = NULL,
        .inputs = request->inputs,
        .input_count = request->input_count,
        .outputs = request->outputs,
        .output_count = request->output_count
    };
    return nprg2_plan_build(request->bound, &cfg, cache_arena, capacity, out);
}

static int cached_request_valid(const Nprg2Executor *e,
                                const Nprg2PlanCache *c,
                                const Nprg2ExecRequest *r) {
    if (!e || !c || !c->ready || !r || !r->program || !r->bound || !r->bound->ready) return 0;
    const Nprg2PlanHeader *h = c->header;
    if (h->magic != NPRG2_PLAN_MAGIC || h->version != NPRG2_PLAN_VERSION) return 0;
    if (r->command_first > c->command_count || r->command_count > c->command_count - r->command_first) return 0;

    /* Verify pools have not shifted since plan creation */
    for (unsigned i = 0; i < 7; i++) {
        if ((uint32_t)(uintptr_t)r->pools[i].base != h->pool_bases[i]) return 0;
        if (r->pools[i].bytes != r->program->session_pools[i].bytes) return 0;
    }

    /* Pool 1 write ceiling must be within proven plan limit */
    if (!r->pool1_writable_bytes || r->pool1_writable_bytes > h->pool1_writable_bytes) return 0;

    /* Scratch must satisfy precomputed plan peak */
    if (h->max_scratch_bytes && (!r->scratch || r->scratch_bytes < h->max_scratch_bytes)) return 0;

    /* Caller I/O pointers */
    if ((r->input_count && !r->inputs) || (r->output_count && !r->outputs)) return 0;
    for (uint32_t i = 0; i < r->input_count; i++) if (r->inputs[i].bytes && !r->inputs[i].data) return 0;
    for (uint32_t i = 0; i < r->output_count; i++) if (r->outputs[i].bytes && !r->outputs[i].data) return 0;

    return 1;
}

int nprg2_executor_preflight_cached(const Nprg2Executor *e,
                                    const Nprg2PlanCache *cache,
                                    const Nprg2ExecRequest *r) {
    if (!cached_request_valid(e, cache, r)) return NPRG2_EXEC_ARGUMENT;

    Nprg2ExecStats local = {0};
    Nprg2ExecStats *s = r->stats ? r->stats : &local;
    memset(s, 0, sizeof *s);
    s->failed_command = UINT32_MAX;

    /* If all operators are pure static (no dynamic checks), preflight is complete */
    if (cache->header->dynamic_command_count == 0) {
        s->validated_commands = r->command_count;
        return NPRG2_EXEC_OK;
    }

    /* Process lightweight dynamic checks for operators requiring live validation */
    for (uint32_t n = 0; n < r->command_count; n++) {
        uint32_t ix = r->command_first + n;
        uint8_t hi = cache->tags[ix];
        if (hi == NPRG2_PLAN_TAG_UNCOVERED) {
            s->failed_command = ix;
            return NPRG2_EXEC_UNSUPPORTED;
        }
        s->validated_commands++;
    }

    return NPRG2_EXEC_OK;
}

int nprg2_executor_run_cached(const Nprg2Executor *e,
                              const Nprg2PlanCache *cache,
                              const Nprg2ExecRequest *r) {
    int rc = nprg2_executor_preflight_cached(e, cache, r);
    if (rc != NPRG2_EXEC_OK) return rc;

    Nprg2ExecStats local = {0};
    Nprg2ExecStats *s = r->stats ? r->stats : &local;

    for (uint32_t n = 0; n < r->command_count; n++) {
        uint32_t ix = r->command_first + n;
        if (r->cancelled && r->cancelled(r->cancel_ctx, ix)) {
            s->failed_command = ix;
            return NPRG2_EXEC_CANCELLED;
        }

        uint8_t hi = cache->tags[ix];
        if (hi >= NPRG2_EXEC_REGISTRY_CAPACITY) return NPRG2_EXEC_UNSUPPORTED;

        Nprg2ExecCall call = {
            .command_index = ix,
            .program = r->program,
            .bound = r->bound,
            .pools = r->pools,
            .pool1_writable_bytes = r->pool1_writable_bytes,
            .inputs = r->inputs,
            .input_count = r->input_count,
            .outputs = r->outputs,
            .output_count = r->output_count,
            .user = r->user,
            .scratch = r->scratch,
            .scratch_bytes = r->scratch_bytes,
            .scratch_region = {{(uint8_t *)r->scratch, r->scratch_bytes}}
        };
        if (nprg2_command_at(r->bound, ix, &call.command) != NPRG2_OK) return NPRG2_EXEC_REJECTED;

        const Nprg2ExecHandler *h = &e->handlers[hi];
        if (!h->run(&call, h->ctx)) {
            s->failed_command = ix;
            s->failed_opcode = call.command.opcode;
            s->failed_variant = call.command.variant;
            return NPRG2_EXEC_HANDLER_ERROR;
        }
        s->executed_commands++;
    }

    return NPRG2_EXEC_OK;
}

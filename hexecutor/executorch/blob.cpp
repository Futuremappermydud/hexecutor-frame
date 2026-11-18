#include "blob.hpp"

#include <cstring>
#include <limits>

namespace hexecutor_et {
namespace {

class Reader {
  public:
    Reader(const uint8_t *data, size_t size) : p_(data), end_(data + size) {}
    bool u32(uint32_t *v) {
        if (remaining() < 4)
            return false;
        *v = uint32_t(p_[0]) | uint32_t(p_[1]) << 8 | uint32_t(p_[2]) << 16 |
             uint32_t(p_[3]) << 24;
        p_ += 4;
        return true;
    }
    bool u64(uint64_t *v) {
        if (remaining() < 8)
            return false;
        uint32_t lo, hi;
        u32(&lo);
        u32(&hi);
        *v = uint64_t(lo) | uint64_t(hi) << 32;
        return true;
    }
    bool bytes(size_t n, std::vector<uint8_t> *out) {
        if (remaining() < n)
            return false;
        out->assign(p_, p_ + n);
        p_ += n;
        return true;
    }
    bool magic(const char m[4]) {
        if (remaining() < 4 || std::memcmp(p_, m, 4) != 0)
            return false;
        p_ += 4;
        return true;
    }
    size_t remaining() const { return size_t(end_ - p_); }

  private:
    const uint8_t *p_;
    const uint8_t *end_;
};

bool fail(std::string *error, const char *msg) {
    if (error)
        *error = msg;
    return false;
}

bool parse_tensor(Reader &r, TensorDesc *t, std::string *error) {
    uint32_t layout = 0, rank = 0;
    if (!r.u32(&layout) || !r.u32(&rank))
        return fail(error, "truncated tensor record");
    if (layout != kLayoutAsIs && layout != kLayoutNchw)
        return fail(error, "unknown tensor layout");
    if (rank < 1 || rank > kMaxRank)
        return fail(error, "tensor rank out of range");
    if (layout == kLayoutNchw && rank != 4)
        return fail(error, "NCHW layout requires rank 4");
    t->layout = layout;
    t->dims.clear();
    uint64_t elements = 1;
    for (uint32_t i = 0; i < rank; i++) {
        uint32_t d = 0;
        if (!r.u32(&d))
            return fail(error, "truncated tensor dims");
        if (d == 0)
            return fail(error, "zero-sized dimension");
        if (elements > std::numeric_limits<uint64_t>::max() / d)
            return fail(error, "tensor element count overflows");
        elements *= d;
        t->dims.push_back(d);
    }
    // Byte size must fit in size_t and be representable as float32 bytes.
    if (elements > std::numeric_limits<uint64_t>::max() / 4 ||
        elements * 4 > uint64_t(std::numeric_limits<size_t>::max()) ||
        elements > uint64_t(std::numeric_limits<int32_t>::max()))
        return fail(error, "tensor byte size overflows");
    t->elements = elements;
    t->bytes = elements * 4;
    return true;
}

} // namespace

bool parse_blob(const uint8_t *data, size_t size, Blob *out, std::string *error) {
    if (!data || !out)
        return fail(error, "null blob");
    Reader r(data, size);
    if (!r.magic("HXET"))
        return fail(error, "bad magic");
    uint32_t version = 0, backend = 0, n_in = 0, n_out = 0;
    if (!r.u32(&version) || !r.u32(&backend) || !r.u32(&n_in) || !r.u32(&n_out))
        return fail(error, "truncated header");
    if (version != kBlobVersion)
        return fail(error, "unsupported blob version");
    if (backend != kBackendTensorF32)
        return fail(error, "unsupported backend");
    if (n_in > kMaxIo || n_out > kMaxIo)
        return fail(error, "too many inputs or outputs");
    Blob b;
    b.backend = backend;
    b.inputs.resize(n_in);
    b.outputs.resize(n_out);
    for (auto &t : b.inputs)
        if (!parse_tensor(r, &t, error))
            return false;
    for (auto &t : b.outputs)
        if (!parse_tensor(r, &t, error))
            return false;
    uint64_t image_size = 0, constants_size = 0;
    if (!r.u64(&image_size) || !r.u64(&constants_size))
        return fail(error, "truncated section sizes");
    if (image_size == 0)
        return fail(error, "empty image");
    if (image_size > r.remaining() || constants_size > r.remaining() - image_size)
        return fail(error, "section sizes exceed blob");
    if (!r.bytes(size_t(image_size), &b.image) || !r.bytes(size_t(constants_size), &b.constants))
        return fail(error, "truncated sections");
    if (r.remaining() != 0)
        return fail(error, "trailing bytes after sections");
    *out = std::move(b);
    return true;
}

void nchw_to_nhwc(const float *src, float *dst, const uint32_t d[4]) {
    const size_t n = d[0], c = d[1], h = d[2], w = d[3];
    for (size_t in = 0; in < n; in++)
        for (size_t ic = 0; ic < c; ic++)
            for (size_t ih = 0; ih < h; ih++)
                for (size_t iw = 0; iw < w; iw++)
                    dst[((in * h + ih) * w + iw) * c + ic] = src[((in * c + ic) * h + ih) * w + iw];
}

void nhwc_to_nchw(const float *src, float *dst, const uint32_t d[4]) {
    const size_t n = d[0], c = d[1], h = d[2], w = d[3];
    for (size_t in = 0; in < n; in++)
        for (size_t ic = 0; ic < c; ic++)
            for (size_t ih = 0; ih < h; ih++)
                for (size_t iw = 0; iw < w; iw++)
                    dst[((in * c + ic) * h + ih) * w + iw] = src[((in * h + ih) * w + iw) * c + ic];
}

} // namespace hexecutor_et

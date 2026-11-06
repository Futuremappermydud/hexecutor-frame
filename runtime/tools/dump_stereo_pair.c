#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

int main() {
    pid_t pid = 59153;
    int pfd = syscall(__NR_pidfd_open, pid, 0);
    if (pfd < 0) { perror("pidfd_open"); return 1; }

    int fds[] = { 189, 197 }; // Right slot 0, Left slot 0
    const char *names[] = { "/home/steamos/test_openvr/cam_right.raw", "/home/steamos/test_openvr/cam_left.raw" };
    size_t size = 9564160;

    for (int i = 0; i < 2; i++) {
        int lfd = syscall(__NR_pidfd_getfd, pfd, fds[i], 0);
        if (lfd < 0) { perror("pidfd_getfd"); continue; }

        void *ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, lfd, 0);
        if (ptr == MAP_FAILED) { perror("mmap"); close(lfd); continue; }

        FILE *fp = fopen(names[i], "wb");
        if (fp) {
            fwrite(ptr, 1, size, fp);
            fclose(fp);
            printf("Saved %s (%zu bytes)\n", names[i], size);
        }
        munmap(ptr, size);
        close(lfd);
    }

    close(pfd);
    return 0;
}

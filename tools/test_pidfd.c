#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <pid> <target_fd>\n", argv[0]);
        return 1;
    }
    pid_t pid = atoi(argv[1]);
    int target_fd = atoi(argv[2]);

    int pfd = syscall(__NR_pidfd_open, pid, 0);
    if (pfd < 0) {
        perror("pidfd_open");
        return 1;
    }

    int my_fd = syscall(__NR_pidfd_getfd, pfd, target_fd, 0);
    if (my_fd < 0) {
        perror("pidfd_getfd");
        close(pfd);
        return 1;
    }

    printf("Successfully acquired FD %d from PID %d as local FD %d!\n", target_fd, pid, my_fd);

    // Try to mmap it
    size_t size = 9564160;
    void *ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, my_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
    } else {
        printf("Successfully mmapped %zu bytes at %p!\n", size, ptr);
        unsigned char *b = (unsigned char *)ptr;
        printf("First 16 bytes: ");
        for (int i = 0; i < 16; i++) printf("%02x ", b[i]);
        printf("\n");
        munmap(ptr, size);
    }

    close(my_fd);
    close(pfd);
    return 0;
}

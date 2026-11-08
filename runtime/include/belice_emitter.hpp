#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace FrameSpatial {

// BELICE V2 emitter on its MCP2221's GP0 through hidraw (low = on). Each command waits
// for the chip's reply, about 2 ms over USB. The chip has no watchdog: GP0 holds its last
// level, so a killed process can leave the emitter on until something sends off.
class BeliceEmitter {
public:
    BeliceEmitter() = default;
    BeliceEmitter(const BeliceEmitter&) = delete;
    BeliceEmitter& operator=(const BeliceEmitter&) = delete;
    ~BeliceEmitter() {
        if (fd_ >= 0) {
            set(false);
            close(fd_);
        }
    }

    bool open() {
        DIR* classes = opendir("/sys/class/hidraw");
        if (!classes) return false;
        std::string device;
        while (dirent* entry = readdir(classes)) {
            if (entry->d_name[0] == '.') continue;
            std::ifstream uevent(std::string("/sys/class/hidraw/") + entry->d_name + "/device/uevent");
            for (std::string line; std::getline(uevent, line);)
                if (line.find("000004D8:000000DD") != std::string::npos) device = std::string("/dev/") + entry->d_name;
        }
        closedir(classes);
        if (device.empty()) return false;
        fd_ = ::open(device.c_str(), O_RDWR);
        return fd_ >= 0 && set(false);
    }

    bool set(bool on) {
        std::array<uint8_t, 65> command{};  // leading 0 is the HID report ID
        command[1] = 0x50;                  // Set GPIO output values
        command[3] = 1;                     // alter GP0
        command[4] = on ? 0 : 1;
        if (write(fd_, command.data(), command.size()) != ssize_t(command.size())) return false;
        pollfd ready{fd_, POLLIN, 0};
        if (poll(&ready, 1, 100) <= 0) return false;
        std::array<uint8_t, 64> reply{};
        return read(fd_, reply.data(), reply.size()) == ssize_t(reply.size()) && reply[0] == 0x50 && reply[1] == 0 &&
               reply[2] != 0xEE;  // 0xEE: GP0 is not in GPIO mode
    }

private:
    int fd_ = -1;
};

}  // namespace FrameSpatial

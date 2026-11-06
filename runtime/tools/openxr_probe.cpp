#include <openxr/openxr.h>

#include <iostream>
#include <vector>

int main() {
    uint32_t count = 0;
    XrResult result = xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
    if (XR_FAILED(result)) {
        std::cerr << "xrEnumerateInstanceExtensionProperties failed: " << result << '\n';
        return 1;
    }
    std::vector<XrExtensionProperties> extensions(
        count, {XR_TYPE_EXTENSION_PROPERTIES});
    result = xrEnumerateInstanceExtensionProperties(
        nullptr, count, &count, extensions.data());
    if (XR_FAILED(result)) return 2;
    for (const auto& extension : extensions) {
        std::cout << extension.extensionName << ' ' << extension.extensionVersion << '\n';
    }
    return 0;
}

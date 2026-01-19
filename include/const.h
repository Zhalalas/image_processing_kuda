#include <array>
#include <map>
#include <string>
using namespace std;

// Sample images path
inline constexpr const char* SAMPLES_PATH = "sample_images/";

// Fixed-size 5-point landmark template (5x2)
using Landmark5 = array<std::array<float, 2>, 5>;

// Source templates (_SRC1 through _SRC5)
inline constexpr Landmark5 SRC1 = {{
    {{51.642f, 50.115f}},
    {{57.617f, 49.990f}},
    {{35.740f, 69.007f}},
    {{51.157f, 89.050f}},
    {{57.025f, 89.702f}}
}};

inline constexpr Landmark5 SRC2 = {{
    {{45.031f, 50.118f}},
    {{65.568f, 50.872f}},
    {{39.677f, 68.111f}},
    {{45.177f, 86.190f}},
    {{64.246f, 86.758f}}
}};

inline constexpr Landmark5 SRC3 = {{
    {{39.730f, 51.138f}},
    {{72.270f, 51.138f}},
    {{56.000f, 68.493f}},
    {{42.463f, 87.010f}},
    {{69.537f, 87.010f}}
}};

inline constexpr Landmark5 SRC4 = {{
    {{46.845f, 50.872f}},
    {{67.382f, 50.118f}},
    {{72.737f, 68.111f}},
    {{48.167f, 86.758f}},
    {{67.236f, 86.190f}}
}};

inline constexpr Landmark5 SRC5 = {{
    {{54.796f, 49.990f}},
    {{60.771f, 50.115f}},
    {{76.673f, 69.007f}},
    {{55.388f, 89.702f}},
    {{61.257f, 89.050f}}
}};

// Combined source templates
inline const array<Landmark5, 5> SRC = {SRC1, SRC2, SRC3, SRC4, SRC5};

// Template mapping for different image sizes
inline const map<int, std::array<Landmark5, 5>> _SRC_MAP = {
    {112, SRC},
    {224, SRC}  // if scaling needed, can be scaled dynamically in code
};

// ArcFace 5-point reference landmarks for 112x112
inline constexpr Landmark5 ARC_FACE_TEMPLATE = {{
    {{38.2946f, 56.0f}},   // left eye
    {{73.5318f, 56.0f}},   // right eye
    {{56.0252f, 76.0f}},   // nose
    {{41.5493f, 96.0f}},   // left mouth
    {{70.7299f, 96.0f}}    // right mouth
}};

// Pre-calculate ArcFace template for 224x224
inline constexpr Landmark5 ARC_FACE_TEMPLATE_224 = {{
    {{(38.2946f - 56.0f) * 2.0f + 112.0f, (56.0f - 56.0f) * 2.0f + 112.0f}},
    {{(73.5318f - 56.0f) * 2.0f + 112.0f, (56.0f - 56.0f) * 2.0f + 112.0f}},
    {{(56.0252f - 56.0f) * 2.0f + 112.0f, (76.0f - 56.0f) * 2.0f + 112.0f}},
    {{(41.5493f - 56.0f) * 2.0f + 112.0f, (96.0f - 56.0f) * 2.0f + 112.0f}},
    {{(70.7299f - 56.0f) * 2.0f + 112.0f, (96.0f - 56.0f) * 2.0f + 112.0f}}
}};

// ArcFace template map
inline const map<int, Landmark5> ARC_FACE_TEMPLATE_MAP = {
    {112, ARC_FACE_TEMPLATE},
    {224, ARC_FACE_TEMPLATE_224}
};

// Template modes
inline const array<const char*, 2> TEMPLATE_MODES = {"arcface", "default"};

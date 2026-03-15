#ifndef OPTIMIZER_METADATA_KEYS_HPP_INCLUDED
#define OPTIMIZER_METADATA_KEYS_HPP_INCLUDED

#include <cstdint>
#include <utility/assumptions.hpp>

#include <string>

namespace optimizer::metadata
{

enum class MetaKey : uint8_t
{
    TRANSLATION_SALA_TO_IR,

    POINTS_TO,
    AVAIL_COPY,
};

inline std::string to_string(MetaKey key)
{
    switch (key)
    {
    case MetaKey::TRANSLATION_SALA_TO_IR:
        return "TRANSLATION_SALA_TO_IR";
    case MetaKey::POINTS_TO:
        return "POINTS_TO";
    case MetaKey::AVAIL_COPY:
        return "AVAIL_COPY";

    default:
        return "<<ERROR>>: undefined for: " + std::to_string(static_cast<int>(key));
    }
}

} // namespace optimizer::metadata

#endif

#ifndef OPTIMIZER_METADATA_HPP_INCLUDED
#define OPTIMIZER_METADATA_HPP_INCLUDED

#include <optimizer/metadata/meta_entry.hpp>
#include <optimizer/metadata/meta_keys.hpp>

#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace optimizer::metadata
{
template <typename T>
concept HasExactMetaKey = requires { T::key; } &&
                          std::same_as<std::remove_cvref_t<decltype(T::key)>, metadata::MetaKey>;

template <typename BaseEntry>
    requires std::derived_from<BaseEntry, MetaEntryI>
class Metadata
{
  private:
    template <typename T>
    static constexpr bool valid_retrievable = std::is_base_of_v<BaseEntry, T> && HasExactMetaKey<T>;

    template <typename T>
    static constexpr bool valid_settable = std::derived_from<T, BaseEntry> && HasExactMetaKey<T>;

  public:
    Metadata() = default;

    template <typename T>
        requires valid_settable<T>
    void set(std::unique_ptr<T> entry)
    {
        data_[T::key] = std::move(entry);
    }

    template <typename T>
        requires valid_retrievable<T>
    T& get()
    {
        auto iter = data_.find(T::key);
        if (iter == data_.end())
        {
            throw std::runtime_error("Metadata key not found: " + to_string(T::key));
        }
        T* casted = dynamic_cast<T*>(iter->second.get());
        if (!casted)
        {
            throw std::runtime_error("Metadata type mismatch for key: " + to_string(T::key));
        }
        return *casted;
    }

    template <typename T>
        requires valid_retrievable<T>
    const T& get() const
    {
        auto iter = data_.find(T::key);
        if (iter == data_.end())
        {
            throw std::runtime_error("Metadata key not found: " + to_string(T::key));
        }
        const T* casted = dynamic_cast<const T*>(iter->second.get());
        if (!casted)
        {
            throw std::runtime_error("Metadata type mismatch for key: " + to_string(T::key));
        }
        return *casted;
    }

    template <typename T>
        requires valid_retrievable<T>
    std::optional<T*> try_get() const
    {
        auto iter = data_.find(T::key);
        if (iter == data_.end())
        {
            return std::nullopt;
        }
        T* casted = dynamic_cast<T*>(iter->second.get());
        return casted ? std::optional<T*>{casted} : std::nullopt;
    }

    template <typename T>
        requires valid_retrievable<T>
    [[nodiscard]] bool has() const
    {
        return data_.find(T::key) != data_.end();
    }

    void remove(const MetaKey& key) { data_.erase(key); }

    void clear() { data_.clear(); }

  private:
    std::unordered_map<MetaKey, std::unique_ptr<BaseEntry>> data_;
};
} // namespace optimizer::metadata

#endif

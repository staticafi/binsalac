#ifndef OPTIMIZER_UTILS_SPARSE_MAP_INCLUDED
#define OPTIMIZER_UTILS_SPARSE_MAP_INCLUDED

#include <algorithm>
#include <vector>
namespace optimizer::utils
{

template <typename KeyT, typename ValueT>
class SparseMap
{
  public:
    using key_type       = KeyT;
    using mapped_type    = ValueT;
    using value_type     = std::pair<key_type, mapped_type>;
    using storage_type   = std::vector<value_type>;
    using iterator       = storage_type::iterator;
    using const_iterator = storage_type::const_iterator;
    using size_type      = storage_type::size_type;

    SparseMap() = default;

    iterator       begin() noexcept { return data_.begin(); }
    iterator       end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    size_type          size() const noexcept { return data_.size(); }
    void               clear() noexcept { data_.clear(); }
    void               reserve(size_type n) { data_.reserve(n); }

    iterator find(const key_type key) noexcept
    {
        auto it = lower_bound_(key);
        if (it != data_.end() && it->first == key)
        {
            return it;
        }
        return data_.end();
    }

    const_iterator find(const key_type key) const noexcept
    {
        auto it = lower_bound_(key);
        if (it != data_.end() && it->first == key)
        {
            return it;
        }
        return data_.end();
    }

    [[nodiscard]] bool contains(const key_type key) const noexcept
    {
        return find(key) != data_.end();
    }

    mapped_type& operator[](const key_type key)
    {
        auto it = lower_bound_(key);
        if (it == data_.end() || it->first != key)
        {
            it = data_.insert(it, value_type{key, mapped_type{}});
        }
        return it->second;
    }

    std::pair<iterator, bool> insert(const value_type& value)
    {
        auto it = lower_bound_(value.first);
        if (it != data_.end() && it->first == value.first)
        {
            return {it, false};
        }
        it = data_.insert(it, value);
        return {it, true};
    }

    std::pair<iterator, bool> insert(value_type&& value)
    {
        auto it = lower_bound_(value.first);
        if (it != data_.end() && it->first == value.first)
        {
            return {it, false};
        }
        it = data_.insert(it, std::move(value));
        return {it, true};
    }

    template <typename K, typename V>
    std::pair<iterator, bool> emplace(K&& key, V&& value)
    {
        return insert(value_type{std::forward<K>(key), std::forward<V>(value)});
    }

    template <typename K, typename V>
    std::pair<iterator, bool> insert_or_assign(K&& key, V&& value)
    {
        auto it = lower_bound_(key);
        if (it != data_.end() && it->first == key)
        {
            it->second = std::forward<V>(value);
            return {it, false};
        }
        it = data_.insert(it, value_type{std::forward<K>(key), std::forward<V>(value)});
        return {it, true};
    }

    void erase(const key_type key)
    {
        if (auto it = find(key); it != data_.end())
        {
            data_.erase(it);
        }
    }

    iterator erase(iterator it) { return data_.erase(it); }

    friend bool operator==(const SparseMap& lhs, const SparseMap& rhs) = default;

  private:
    storage_type data_{};

    iterator lower_bound_(const key_type key) noexcept
    {
        return std::lower_bound(data_.begin(), data_.end(), key,
                                [](const value_type& elem, const key_type k)
                                { return elem.first < k; });
    }

    const_iterator lower_bound_(const key_type key) const noexcept
    {
        return std::lower_bound(data_.begin(), data_.end(), key,
                                [](const value_type& elem, const key_type k)
                                { return elem.first < k; });
    }
};

} // namespace optimizer::utils
#endif

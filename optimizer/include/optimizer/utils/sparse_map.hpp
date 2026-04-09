#ifndef OPTIMIZER_UTILS_SPARSE_MAP_INCLUDED
#define OPTIMIZER_UTILS_SPARSE_MAP_INCLUDED

#include <algorithm>
#include <stdexcept>
#include <utility>
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
    using iterator       = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;
    using size_type      = typename storage_type::size_type;

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

    mapped_type& at(const key_type key)
    {
        auto it = find(key);
        if (it == data_.end())
        {
            throw std::out_of_range("SparseMap::at: key not found");
        }
        return it->second;
    }

    const mapped_type& at(const key_type key) const
    {
        auto it = find(key);
        if (it == data_.end())
        {
            throw std::out_of_range("SparseMap::at: key not found");
        }
        return it->second;
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

    iterator insert(const_iterator hint, const value_type& value)
    {
        return insert_with_hint_(hint, value);
    }

    iterator insert(const_iterator hint, value_type&& value)
    {
        return insert_with_hint_(hint, std::move(value));
    }

    template <typename K, typename V>
    std::pair<iterator, bool> emplace(K&& key, V&& value)
    {
        return insert(value_type{std::forward<K>(key), std::forward<V>(value)});
    }

    template <typename K, typename V>
    iterator emplace_hint(const_iterator hint, K&& key, V&& value)
    {
        return insert(hint, value_type{std::forward<K>(key), std::forward<V>(value)});
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

    template <typename K, typename V>
    iterator insert_or_assign(const_iterator hint, K&& key, V&& value)
    {
        if (data_.empty())
        {
            return data_.insert(data_.begin(),
                                value_type{std::forward<K>(key), std::forward<V>(value)});
        }

        if (hint == data_.end())
        {
            if (data_.back().first < key)
            {
                return data_.insert(data_.end(),
                                    value_type{std::forward<K>(key), std::forward<V>(value)});
            }
            if (data_.back().first == key)
            {
                auto it    = std::prev(data_.end());
                it->second = std::forward<V>(value);
                return it;
            }
        }
        else
        {
            auto idx = static_cast<typename storage_type::difference_type>(
                    std::distance(data_.cbegin(), hint));
            auto it = data_.begin() + idx;

            if (it->first == key)
            {
                it->second = std::forward<V>(value);
                return it;
            }

            const bool ok_after_prev = (it == data_.begin()) || (std::prev(it)->first < key);
            const bool ok_before_it  = key < it->first;

            if (ok_after_prev && ok_before_it)
            {
                return data_.insert(it, value_type{std::forward<K>(key), std::forward<V>(value)});
            }

            if (it != data_.begin() && std::prev(it)->first == key)
            {
                auto prev    = std::prev(it);
                prev->second = std::forward<V>(value);
                return prev;
            }
        }

        return insert_or_assign(std::forward<K>(key), std::forward<V>(value)).first;
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

    template <typename U>
    iterator insert_with_hint_(const_iterator hint, U&& value)
    {
        const key_type& key = value.first;

        if (data_.empty())
        {
            return data_.insert(data_.begin(), std::forward<U>(value));
        }

        if (hint == data_.end())
        {
            if (data_.back().first < key)
            {
                return data_.insert(data_.end(), std::forward<U>(value));
            }
            if (data_.back().first == key)
            {
                return std::prev(data_.end());
            }
            return insert(std::forward<U>(value)).first;
        }

        auto idx = static_cast<typename storage_type::difference_type>(
                std::distance(data_.cbegin(), hint));
        auto it = data_.begin() + idx;

        if (it->first == key)
        {
            return it;
        }

        const bool ok_after_prev = (it == data_.begin()) || (std::prev(it)->first < key);
        const bool ok_before_it  = key < it->first;

        if (ok_after_prev && ok_before_it)
        {
            return data_.insert(it, std::forward<U>(value));
        }

        if (it != data_.begin() && std::prev(it)->first == key)
        {
            return std::prev(it);
        }

        return insert(std::forward<U>(value)).first;
    }
};

} // namespace optimizer::utils

#endif

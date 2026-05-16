#ifndef OPTIMIZER_UTILS_SPARSE_SET_HPP_INCLUDED
#define OPTIMIZER_UTILS_SPARSE_SET_HPP_INCLUDED

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace optimizer::utils
{

template <typename T, typename Compare = std::less<T>>
class SparseSet
{
  public:
    using value_type     = T;
    using storage_type   = std::vector<value_type>;
    using iterator       = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;
    using size_type      = typename storage_type::size_type;

    SparseSet() = default;

    SparseSet(std::initializer_list<value_type> init) { insert(init.begin(), init.end()); }

    template <typename It>
    SparseSet(It first, It last)
    {
        insert(first, last);
    }

    iterator       begin() noexcept { return data_.begin(); }
    iterator       end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    [[nodiscard]] bool      empty() const noexcept { return data_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }

    void clear() noexcept { data_.clear(); }
    void reserve(size_type n) { data_.reserve(n); }

    [[nodiscard]] bool contains(const value_type& value) const noexcept
    {
        return find(value) != data_.end();
    }

    const_iterator find(const value_type& value) const noexcept
    {
        auto it = lower_bound_(value);
        if (it != data_.end() && equal_(*it, value))
        {
            return it;
        }
        return data_.end();
    }

    iterator find(const value_type& value) noexcept
    {
        auto it = lower_bound_(value);
        if (it != data_.end() && equal_(*it, value))
        {
            return it;
        }
        return data_.end();
    }

    std::pair<iterator, bool> insert(const value_type& value)
    {
        auto it = lower_bound_(value);
        if (it != data_.end() && equal_(*it, value))
        {
            return {it, false};
        }
        it = data_.insert(it, value);
        return {it, true};
    }

    std::pair<iterator, bool> insert(value_type&& value)
    {
        auto it = lower_bound_(value);
        if (it != data_.end() && equal_(*it, value))
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

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args)
    {
        value_type value{std::forward<Args>(args)...};
        return insert(std::move(value));
    }

    template <typename... Args>
    iterator emplace_hint(const_iterator hint, Args&&... args)
    {
        value_type value{std::forward<Args>(args)...};
        return insert(hint, std::move(value));
    }

    template <typename It>
    void insert(It first, It last)
    {
        for (; first != last; ++first)
        {
            insert(*first);
        }
    }

    void insert(std::initializer_list<value_type> init) { insert(init.begin(), init.end()); }

    size_type erase(const value_type& value)
    {
        auto it = find(value);
        if (it == data_.end())
        {
            return 0;
        }
        data_.erase(it);
        return 1;
    }

    iterator erase(iterator it) { return data_.erase(it); }

    void join_with(const SparseSet& other) { union_with(other); }

    [[nodiscard]] friend bool operator==(const SparseSet& lhs, const SparseSet& rhs) noexcept
    {
        return lhs.data_ == rhs.data_;
    }

    [[nodiscard]] friend auto operator<=>(const SparseSet& lhs, const SparseSet& rhs) noexcept
    {
        return lhs.data_ <=> rhs.data_;
    }

    void union_with(const SparseSet& other)
    {
        if (this == &other || other.empty())
        {
            return;
        }
        if (empty())
        {
            data_ = other.data_;
            return;
        }
        if (data_ == other.data_)
        {
            return;
        }

        // all current elements are strictly before other
        if (cmp_(data_.back(), other.data_.front()))
        {
            data_.reserve(data_.size() + other.data_.size());
            data_.insert(data_.end(), other.data_.begin(), other.data_.end());
            return;
        }

        // all other elements are strictly before current
        if (cmp_(other.data_.back(), data_.front()))
        {
            storage_type merged;
            merged.reserve(data_.size() + other.data_.size());
            merged.insert(merged.end(), other.data_.begin(), other.data_.end());
            merged.insert(merged.end(), data_.begin(), data_.end());
            data_.swap(merged);
            return;
        }

        storage_type merged;
        merged.reserve(data_.size() + other.data_.size());

        auto lhs = data_.begin();
        auto rhs = other.data_.begin();

        while (lhs != data_.end() && rhs != other.data_.end())
        {
            if (cmp_(*lhs, *rhs))
            {
                merged.push_back(*lhs);
                ++lhs;
            }
            else if (cmp_(*rhs, *lhs))
            {
                merged.push_back(*rhs);
                ++rhs;
            }
            else
            {
                merged.push_back(*lhs);
                ++lhs;
                ++rhs;
            }
        }

        merged.insert(merged.end(), lhs, data_.end());
        merged.insert(merged.end(), rhs, other.data_.end());

        data_.swap(merged);
    }

    void intersect_with(const SparseSet& other)
    {
        if (empty())
        {
            return;
        }
        if (other.empty())
        {
            clear();
            return;
        }

        storage_type kept;
        kept.reserve((std::min)(data_.size(), other.data_.size()));

        auto lhs = data_.begin();
        auto rhs = other.data_.begin();

        while (lhs != data_.end() && rhs != other.data_.end())
        {
            if (cmp_(*lhs, *rhs))
            {
                ++lhs;
            }
            else if (cmp_(*rhs, *lhs))
            {
                ++rhs;
            }
            else
            {
                kept.push_back(*lhs);
                ++lhs;
                ++rhs;
            }
        }

        data_.swap(kept);
    }

  private:
    storage_type data_{};
    Compare      cmp_{};

    [[nodiscard]] bool equal_(const value_type& lhs, const value_type& rhs) const
    {
        return !cmp_(lhs, rhs) && !cmp_(rhs, lhs);
    }

    iterator lower_bound_(const value_type& value) noexcept
    {
        return std::lower_bound(data_.begin(), data_.end(), value, cmp_);
    }

    const_iterator lower_bound_(const value_type& value) const noexcept
    {
        return std::lower_bound(data_.begin(), data_.end(), value, cmp_);
    }

    template <typename U>
    iterator insert_with_hint_(const_iterator hint, U&& value)
    {
        if (data_.empty())
        {
            return data_.insert(data_.begin(), std::forward<U>(value));
        }

        if (hint == data_.end())
        {
            if (cmp_(data_.back(), value))
            {
                return data_.insert(data_.end(), std::forward<U>(value));
            }
            if (equal_(data_.back(), value))
            {
                return std::prev(data_.end());
            }
            return insert(std::forward<U>(value)).first;
        }

        if (equal_(*hint, value))
        {
            return data_.begin() + static_cast<typename storage_type::difference_type>(
                                           std::distance(data_.cbegin(), hint));
        }

        const bool ok_after_prev  = (hint == data_.begin()) || cmp_(*std::prev(hint), value);
        const bool ok_before_hint = cmp_(value, *hint);

        if (ok_after_prev && ok_before_hint)
        {
            return data_.insert(hint, std::forward<U>(value));
        }

        if (hint != data_.begin() && equal_(*std::prev(hint), value))
        {
            return data_.begin() + static_cast<typename storage_type::difference_type>(
                                           std::distance(data_.cbegin(), std::prev(hint)));
        }

        return insert(std::forward<U>(value)).first;
    }
};

} // namespace optimizer::utils

#endif

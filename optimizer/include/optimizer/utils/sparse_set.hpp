#ifndef OPTIMIZER_UTILS_SPARSE_SET_HPP_INCLUDED
#define OPTIMIZER_UTILS_SPARSE_SET_HPP_INCLUDED

#include <algorithm>
#include <initializer_list>
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

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args)
    {
        value_type value{std::forward<Args>(args)...};
        return insert(std::move(value));
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
        if (other.empty())
        {
            return;
        }
        if (empty())
        {
            data_ = other.data_;
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
};

} // namespace optimizer::utils

#endif

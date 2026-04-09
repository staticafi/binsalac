#include <optimizer/utils/type_list.hpp>

using namespace optimizer::utils;

namespace sanity
{
using A = TypeList<>;
using B = TypeList<int, double>;
using C = TypeList<int, float>;

static_assert(contains_v<int, B>);
static_assert(!contains_v<char, B>);

using U = set_union_unique_t<A, B>; // TypeList<int, double>
using I = set_intersection_t<B, C>; // TypeList<int>

static_assert(std::is_same_v<U, TypeList<int, double>>);
static_assert(std::is_same_v<I, TypeList<int>>);

static_assert(is_subset_v<TypeList<>, B>);
static_assert(is_subset_v<TypeList<int>, B>);
static_assert(!is_subset_v<TypeList<float>, B>);

}; // namespace sanity

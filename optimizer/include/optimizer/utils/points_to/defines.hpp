#ifndef OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED
#define OPTIMIZER_UTILS_POINTS_TO_DEFINES_HPP_INCLUDED
#include <iostream>
#include <optimizer/programIR/instruction_ir.hpp>

#include <cstdint>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
namespace optimizer::utils::points_to
{
using objectId   = std::int32_t;
using offsetFlag = bool;

// Grouped objects/memory regions where we do not separate concrete objects
namespace grouped_objects
{
// No known information about object, access nukes the may and must points to sets
constexpr objectId UNDEFINED = -1;

// Access to this propagates this node through objects reachable via global scope ??
constexpr objectId UNKOWN_GLOBAL = -2;

// Not a local variable or a parameter in a specific function context
constexpr objectId OUT_OF_LOCAL_SCOPE = -3;

// Local variable or a pateremter in a specific function context
constexpr objectId OUT_OF_GLOBAL_SCOPE = -4;

// Join of two predecessors block where one defines may points to set for var X and one does not
constexpr objectId MERGE_UNKNOWN = -5;

// One function initializes  points to information for a STATIC VARIABLE while atleast one other
// does not, abstract object representing  that MAY POINTS to information differs based on call
// order
constexpr objectId CALL_ORDER_DISCREPENCY = -6;

// Functions
constexpr objectId FUNCTION = -7;

// Vargarg block
constexpr objectId VARGARG_BLOCK = -8;

// Memory gained via ALLOCA
constexpr objectId ALLOCA = -9;

// Memory obtained via HEAP
constexpr objectId HEAP = -10;
} // namespace grouped_objects

// Region of the memory where the variable lives
enum class RegionTag : std::uint8_t
{
    // Local variables
    Local,

    // Parameters
    Parameter,

    // Static variables
    Static,

    // Constants
    Constant,
};

static inline std::ostream& operator<<(std::ostream& os, const RegionTag region)
{
    switch (region)
    {
    case RegionTag::Local:
        return os << "local";
    case RegionTag::Parameter:
        return os << "param";
    case RegionTag::Static:
        return os << "static";
    case RegionTag::Constant:
        return os << "const";
    default:
        return os << static_cast<int>(region);
    }
}

struct Object
{
    objectId  id{grouped_objects::UNDEFINED};
    RegionTag region{};
};

struct Target
{
    objectId id;
    bool     offset_flag{false};

    friend bool operator==(const Target& a, const Target& b)
    {
        return a.id == b.id && a.offset_flag == b.offset_flag;
    }
};

static inline std::ostream& operator<<(std::ostream& os, const Target& object)
{
    os << object.id;
    if (object.offset_flag)
    {
        os << "off";
    }
    return os;
}

struct TargetHash
{
    std::size_t operator()(const Target& target) const noexcept
    {
        const auto h1 = std::hash<objectId>{}(target.id);
        const auto h2 = std::hash<bool>{}(target.offset_flag);

        // TODO: check this
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
    }
};

// Pool of concrete objects
using ObjectPool = std::unordered_map<objectId, Object>;

// Mays set
using MaySet = std::unordered_set<Target, TargetHash>;
// May state
using MayState = std::unordered_map<objectId, MaySet>;

// Must state
using MustState = std::unordered_map<objectId, Target>;

struct ProgramPoint
{
    std::size_t function;
    std::size_t bb;
    std::size_t instr;
};

struct MayTransferContextBundle
{
    ProgramPoint                 pp;
    sala::Instruction::Opcode    opcode;
    MayState&                    may_in;
    const MustState&             must_in;
    const ObjectPool&            global_objects;
    const ObjectPool&            local_objects;
    const std::vector<objectId>& operands_id;
    const std::size_t            operands_count;
    const objectId               last_local_id;
};

struct MustTransferContextBundle
{
    sala::Instruction::Opcode    opcode;
    MustState&                   must_in;
    const MayState&              may_in;
    const ObjectPool&            global_objects;
    const ObjectPool&            local_objects;
    const std::vector<objectId>& operands_id;
    const std::size_t            operands_count;
    const objectId               last_local_id;
};

static inline bool contains_only_objectId(const MaySet& may, objectId id)
{
    // with and without offset
    if (may.size() > 2)
    {
        return false;
    }

    for (const auto target : may)
    {
        if (target.id != id)
        {
            return false;
        }
    }

    return true;
}

static inline bool
is_objectId_reachable(const MayTransferContextBundle& context, objectId source, objectId target,
                      std::size_t max_depth = std::numeric_limits<std::size_t>::max())
{
    if (source == target)
    {
        return true;
    }

    std::unordered_set<objectId>                 seen;
    std::queue<std::pair<objectId, std::size_t>> wl;

    seen.insert(source);
    wl.emplace(source, 0); // depth 0

    while (!wl.empty())
    {
        const auto [current, depth] = wl.front();
        wl.pop();

        if (depth >= max_depth)
        {
            continue; // do not expand further
        }

        const auto it = context.may_in.find(current);
        if (it == context.may_in.end())
        {
            continue;
        }

        for (const auto& next : it->second)
        {
            const objectId nid = next.id;

            if (nid == target)
            {
                return true;
            }

            if (seen.insert(nid).second)
            {
                wl.emplace(nid, depth + 1);
            }
        }
    }

    return false;
}

static inline bool contains_objectId(const MaySet& may, objectId id)
{
    return may.contains({id, false}) || may.contains({id, true});
}

static inline void state_join_or_relaxed(MayState& A, const MayState& B)
{

    for (const auto& [source, transfer] : B)
    {
        auto A_source_iter = A.find(source);
        if (A_source_iter == A.end())
        {
            A_source_iter = A.emplace(source, transfer).first;
        }
        else
        {
            for (const auto transfer_elem : transfer)
            {
                A_source_iter->second.insert(transfer_elem);
            }
        }
    }
}

static inline void state_join_or_strict(MayState& A, const MayState& B,
                                        objectId extension_node = grouped_objects::MERGE_UNKNOWN)
{
    for (auto& [target, pointees] : A)
    {
        if (!B.contains(target))
        {
            pointees.insert({extension_node, false});
        }
    }

    for (const auto& [source, transfer] : B)
    {
        auto source_iter = A.find(source);
        if (source_iter == A.end())
        {
            source_iter = A.insert_or_assign(source, MaySet{{extension_node, false}}).first;
        }
        for (const auto elem : transfer)
        {
            source_iter->second.insert(elem);
        }
    }
}

static inline void state_join_and(MustState& A, MustState const& B)
{
    for (auto A_kvp_iter = A.begin(); A_kvp_iter != A.end();)
    {
        const auto A_kvp_iter_next = std::next(A_kvp_iter);
        if (const auto B_kvp_iter = B.find(A_kvp_iter->first); B_kvp_iter != B.end())
        {
            if (B_kvp_iter->second != A_kvp_iter->second)
            {
                A.erase(A_kvp_iter);
            }
        }
        else
        {
            A.erase(A_kvp_iter);
        }
        A_kvp_iter = A_kvp_iter_next;
    }
}

static inline void transitive_kill(MustState& must_in, const int id, const MayState& may)
{
    std::queue<objectId> wl;
    wl.push(id);
    while (!wl.empty())
    {
        const auto to_kill = wl.front();
        wl.pop();

        const auto to_kill_may_iter = may.find(to_kill);
        if (to_kill_may_iter == may.end())
        {
            continue;
        }

        for (const auto children : to_kill_may_iter->second)
        {
            wl.push(children.id);
        }
        must_in.erase(to_kill);
    }
}

static inline void dump_may_set(const MayState& may_in)
{
    for (const auto kvp : may_in)
    {
        std::cout << kvp.first << ": { ";
        for (const auto target : kvp.second)
        {
            std::cout << target << " ";
        }
        std::cout << "}\n";
    }
}

static void nuke_must(MustState& must_state)
{
    must_state.clear();
}

static void nuke_may(const MayTransferContextBundle& context)
{
    // TODO: REMOVE dump
    // dump_may_set(context);
    for (auto& [_, may_set] : context.may_in)
    {
        may_set = {{grouped_objects::UNDEFINED, false}};
    }
}

static void nuke_reachable_must(objectId source, const MustTransferContextBundle& context)
{
    std::unordered_set<objectId> seen;
    const auto                   source_may_iter = context.may_in.find(source);
    if (source_may_iter == context.may_in.end())
    {
        return;
    }

    std::queue<objectId> wl;
    // first indirection for reachable targets
    for (const auto target : source_may_iter->second)
    {
        if (!seen.contains(target.id))
        {
            wl.push(target.id);
            seen.insert(target.id);
        }
    }

    // deletes currents information
    while (!wl.empty())
    {
        const auto current = wl.front();
        wl.pop();
        const auto current_may_iter = context.may_in.find(current);
        if (current_may_iter == context.may_in.end())
        {
            continue;
        }

        for (const auto target : current_may_iter->second)
        {
            if (!seen.contains(target.id))
            {
                wl.push(target.id);
                seen.insert(target.id);
            }
        }
        context.must_in.erase(current);
    }
}

static void nuke_reachable_may(objectId source, const MayTransferContextBundle& context,
                               objectId result                 = grouped_objects::UNDEFINED,
                               bool     unmodifiable_constants = true)
{
    std::unordered_set<objectId> seen;
    const auto                   source_may_iter = context.may_in.find(source);
    if (source_may_iter == context.may_in.end())
    {
        return;
    }

    const auto is_constant = [&](objectId id)
    {
        const auto global_object_iter = context.global_objects.find(id);
        return global_object_iter != context.global_objects.end() &&
               global_object_iter->second.region == utils::points_to::RegionTag::Constant;
    };

    std::queue<objectId> wl;
    // first indirection for reachable targets
    for (const auto target : source_may_iter->second)
    {
        if (seen.contains(target.id) || (unmodifiable_constants && is_constant(target.id)))
        {
            continue;
        }
        wl.push(target.id);
        seen.insert(target.id);
    }

    while (!wl.empty())
    {
        const auto current_id = wl.front();
        wl.pop();
        const auto current_may_iter = context.may_in.find(current_id);
        if (current_may_iter == context.may_in.end())
        {
            context.may_in[current_id] = {{result, false}};
            continue;
        }

        for (const auto target : current_may_iter->second)
        {
            if (seen.contains(target.id) || (unmodifiable_constants && is_constant(target.id)))
            {
                continue;
            }
            wl.push(target.id);
            seen.insert(target.id);
        }

        current_may_iter->second.insert({result, false});
    }
}

constexpr static inline bool is_abstract(objectId node)
{
    return node < 0;
};

static inline std::size_t get_relevant_operands_count(const program::InstructionIR& instruction)
{
    switch (instruction.get_opcode())
    {
    case sala::Instruction::Opcode::NOP:
    case sala::Instruction::Opcode::HALT:
    case sala::Instruction::Opcode::__INVALID__:
    case sala::Instruction::Opcode::JUMP:
    case sala::Instruction::Opcode::BRANCH:
    case sala::Instruction::Opcode::RET:
    case sala::Instruction::Opcode::STACKRESTORE:
        return 0;
    case sala::Instruction::Opcode::ADDRESS:
    case sala::Instruction::Opcode::LOAD:
    case sala::Instruction::Opcode::STORE:
    case sala::Instruction::Opcode::COPY:
    case sala::Instruction::Opcode::P2I:
    case sala::Instruction::Opcode::I2P:
        return 2;
    case sala::Instruction::Opcode::MEMCPY:
    case sala::Instruction::Opcode::MEMMOVE:
    case sala::Instruction::Opcode::MOVEPTR:
    case sala::Instruction::Opcode::MEMSET:
        return 2;
    case sala::Instruction::Opcode::VA_START:
    case sala::Instruction::Opcode::VA_END:
    case sala::Instruction::Opcode::VA_ARG:
    case sala::Instruction::Opcode::VA_COPY:
        // TODO: look into these above VA
    case sala::Instruction::Opcode::ALLOCA:
    case sala::Instruction::Opcode::ADD:
    case sala::Instruction::Opcode::SUB:
    case sala::Instruction::Opcode::MUL:
    case sala::Instruction::Opcode::DIV:
    case sala::Instruction::Opcode::REM:
    case sala::Instruction::Opcode::AND:
    case sala::Instruction::Opcode::OR:
    case sala::Instruction::Opcode::XOR:
    case sala::Instruction::Opcode::SHL:
    case sala::Instruction::Opcode::SHR:
    case sala::Instruction::Opcode::NEG:
    case sala::Instruction::Opcode::EXTEND:
    case sala::Instruction::Opcode::TRUNCATE:
    case sala::Instruction::Opcode::F2I:
    case sala::Instruction::Opcode::I2F:
    case sala::Instruction::Opcode::LESS:
    case sala::Instruction::Opcode::LESS_EQUAL:
    case sala::Instruction::Opcode::GREATER:
    case sala::Instruction::Opcode::GREATER_EQUAL:
    case sala::Instruction::Opcode::EQUAL:
    case sala::Instruction::Opcode::UNEQUAL:
    case sala::Instruction::Opcode::ISNAN:
    case sala::Instruction::Opcode::MALLOC:
        return 1;
    case sala::Instruction::Opcode::STACKSAVE:
    case sala::Instruction::Opcode::CALL:
        return instruction.get_operands().size();
    case sala::Instruction::Opcode::FREE:
        return 0;
    default:
        UNREACHABLE();
    }
}
} // namespace optimizer::utils::points_to

#endif

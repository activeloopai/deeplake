#pragma once
#include <base/memory_buffer.hpp>
#include <format/buffer.hpp>

#include <algorithm>
#include <cctype>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

namespace icm {
template <typename ValueType>
class value_traits;

template <>
class value_traits<int32_t>
{
public:
    static constexpr int32_t null_value = -1;

    bool static constexpr is_null(int32_t value)
    {
        return value == null_value;
    }
};

template <typename CharT>
class char_traits;

template <>
class char_traits<char>
{
public:
    static constexpr bool has_transform = true;
#ifdef _MSC_VER
    static char transform(char c)
#else
    static constexpr char transform(char c)
#endif
    {
        return std::tolower(c);
    }
};

template <typename ValueType, typename CharT, typename ValueTraits, typename KeyTraits>
requires(sizeof(CharT) <= 2)
class trie_node
{

public:
    using child_size_type = std::conditional_t<sizeof(CharT) == 1, uint8_t, uint16_t>;
    trie_node()
        : value_(ValueTraits::null_value)
    {
    }

    const ValueType& value() const
    {
        return value_;
    }

    void set_value(const ValueType& value)
    {
        value_ = value;
    }

    /**
     * @brief Get the child node for the given character. creates one if it does not exist.
     *
     * @param c The character to search for.
     * @return The child node for the given character. and bool indicating if the node was created.
     */
    std::pair<trie_node*, bool> try_get_child(CharT c)
    {
        if constexpr (KeyTraits::has_transform) {
            c = KeyTraits::transform(c);
        }
        auto it = std::ranges::find_if(children, [c](const auto& p) {
            return p.first == c;
        });
        if (it == children.end()) {
            auto& new_node = children.emplace_back(c, std::make_unique<trie_node>());
            return {new_node.second.get(), true};
        }
        return {it->second.get(), false};
    }

    ValueType search(const std::basic_string_view<CharT> key) const
    {
        const trie_node* node = this;
        for (char c : key) {
            if constexpr (KeyTraits::has_transform) {
                c = KeyTraits::transform(c);
            }
            auto it = std::ranges::find_if(node->children, [c](const auto& p) {
                return p.first == c;
            });
            if (it == node->children.end()) {
                return ValueTraits::null_value;
            }
            node = it->second.get();
        }
        return node->value_;
    }

public: // accessors for serializable
    std::vector<std::pair<CharT, std::unique_ptr<trie_node>>>& get_children()
    {
        return children;
    }

    ValueType& get_value()
    {
        return value_;
    }

private:
    std::vector<std::pair<CharT, std::unique_ptr<trie_node>>> children;
    ValueType value_;
};
} // namespace icm

#pragma once

#include "trie_node.hpp"

#include <fstream>
#include <memory>
#include <tuple>

namespace icm {
template <typename ValueType, typename CharT = char, typename ValueTraits = value_traits<ValueType>,
          typename KeyTraits = char_traits<CharT>>
requires(sizeof(CharT) <= 2)
class trie
{
    using node_type = trie_node<ValueType, CharT, ValueTraits, KeyTraits>;

public:
    using child_size_type = typename node_type::child_size_type;

    trie()
        : root_(std::make_unique<node_type>()),
          total_size_(sizeof(ValueType) + sizeof(child_size_type) + sizeof(int32_t /*checksum*/))
    {
    }

    void insert(const std::basic_string_view<CharT> key, ValueType value)
    {
        node_type* node = root_.get();
        bool created = false;
        for (char c : key) {
            std::tie(node, created) = node->try_get_child(c);
            if (created) {
                total_size_ += sizeof(CharT) + sizeof(child_size_type) + sizeof(ValueType);
            }
        }
        node->set_value(value);
    }

    ValueType search(const std::basic_string_view<CharT>& key) const
    {
        return root_->search(key);
    }

    size_t get_total_size() const
    {
        return total_size_;
    }

public: // accessors for serializable
    const std::unique_ptr<node_type>& get_root() const
    {
        return root_;
    }

    void set_total_size(size_t sz)
    {
        total_size_ = sz;
    }

private:
    std::unique_ptr<node_type> root_;
    size_t total_size_;
};

} // namespace icm

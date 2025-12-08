#pragma once

#include "impl/const_json_utils.hpp"
#include "exceptions.hpp"
#include "string_map.hpp"

#include <base/assert.hpp>
#include <base/f16.hpp>
#include <base/type_traits.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/reader.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/iterator/transform_iterator.hpp>

#include <istream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace icm {

class const_json
{
private:
    using rapid_const_iter = rapidjson::Value::ConstMemberIterator;
    using rapid_array_const_iterator = rapidjson::Value::ConstValueIterator;

    struct processor
    {
        std::shared_ptr<rapidjson::Document> doc_;
        explicit processor(std::shared_ptr<rapidjson::Document> doc) 
            : doc_(std::move(doc))
        {
        }

        std::pair<std::string_view, const_json> operator()(const rapid_const_iter::value_type& value) const
        {
            return {
                std::string_view(value.name.GetString(), value.name.GetStringLength()),
                const_json(value.value, doc_)
            };
        }

        const_json operator()(const rapidjson::Value& value) const
        {
            return const_json(value, doc_);
        }
    };

    auto make_iterator(const rapid_const_iter& it) const
    {
        return boost::make_transform_iterator(it, processor(doc_));
    }

    auto make_iterator(const rapid_array_const_iterator& it) const
    {
        return boost::make_transform_iterator(it, processor(doc_));
    }

public:
    using const_iterator = boost::transform_iterator<
        processor,
        rapid_const_iter,
        std::pair<std::string_view, const_json>
    >;

    using array_const_iterator = boost::transform_iterator<
        processor,
        rapid_array_const_iterator,
        const_json
    >;

    /// @name Constructors
    /// @{
public:
    const_json()
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetObject();
    }

    explicit const_json(const std::string& v)
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetObject();
        doc_->SetString(v.data(), v.size(), doc_->GetAllocator());
    }

    explicit const_json(std::string_view v)
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetObject();
        doc_->SetString(v.data(), v.size(), doc_->GetAllocator());
    }

    template<typename T>
    requires (base::arithmetic<T>)
    explicit const_json(T v)
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetObject();
        if constexpr (std::is_same_v<T, int32_t>) {
            doc_->SetInt(v);
        }
        if constexpr (std::is_same_v<T, uint32_t>) {
            doc_->SetUint(v);
        }
        if constexpr (std::is_same_v<T, int64_t>) {
            doc_->SetInt64(v);
        }
        if constexpr (std::is_same_v<T, uint64_t>) {
            doc_->SetUint64(v);
        }
        if constexpr (std::is_same_v<T, double>) {
            doc_->SetDouble(v);
        }
        if constexpr (std::is_same_v<T, float>) {
            doc_->SetFloat(v);
        }
        if constexpr (std::is_same_v<T, bool>) {
            doc_->SetBool(v);
        }
        if constexpr (std::is_same_v<T, base::bfloat16> || std::is_same_v<T, base::half>) {
            doc_->SetFloat(static_cast<float>(v));
        }
    }

    explicit const_json(std::vector<const_json>&& v)
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetArray();
        for (auto&& i : v) {
            rapidjson::Value value_copy;
            value_copy.CopyFrom(*i.doc_, doc_->GetAllocator());
            doc_->PushBack(std::move(value_copy), doc_->GetAllocator());
        }
    }

    explicit const_json(icm::string_map<const_json>&& v)
        : doc_(std::make_shared<rapidjson::Document>())
        , val_(&*doc_)
    {
        doc_->SetObject();
        for (auto&& [key, val] : v) {
            rapidjson::Value key_value(key.c_str(), doc_->GetAllocator());
            rapidjson::Value value_copy;
            value_copy.CopyFrom(*val.doc_, doc_->GetAllocator());
            doc_->AddMember(std::move(key_value), std::move(value_copy), doc_->GetAllocator());
        }
    }

    const_json(const const_json& o) noexcept
        : doc_(o.doc_)
        , val_(o.val_ == o.doc_.get() ? &*doc_ : o.val_)
    {
    }

    const_json(const_json&& o) noexcept
        : doc_(std::move(o.doc_))
        , val_(o.val_ == o.doc_.get() ? &*doc_ : o.val_)
    {
    }

    const_json& operator=(const const_json& o) noexcept
    {
        if (this == &o) [[unlikely]] {
            return *this;
        }
        doc_ = o.doc_;
        val_ = (o.val_ == o.doc_.get() ? &*doc_ : o.val_);
        return *this;
    }

    const_json& operator=(const_json&& o) noexcept
    {
        if (this == &o) [[unlikely]] {
            return *this;
        }
        doc_ = std::move(o.doc_);
        val_ = (o.val_ == o.doc_.get() ? &*doc_ : o.val_);
        return *this;
    }

    static const_json null_json()
    {
        const_json j{};
        j.doc_->SetNull();
        return j;
    }

private:
    const_json(const rapidjson::Value& v, std::shared_ptr<rapidjson::Document> d)
        : doc_(std::move(d))
        , val_(&v)
    {
    }
    /// @}

    /// @name Static methods
    /// @{
public:
    template <typename T>
    static const_json parse(T&& i)
    {
        const_json res;
        res.parse_internal(std::forward<T>(i));
        if (res.doc_->HasParseError()) {
            throw icm::exception("Failed to parse JSON");
        }
        return res;
    }

    static std::vector<std::string> keys(std::string_view json_content)
    {
        impl::key_collector handler;
        rapidjson::Reader reader;
        rapidjson::MemoryStream stream(json_content.data(), json_content.size());
        reader.Parse(stream, handler);

        return std::move(handler.keys);
    }
    /// @}

    /// @name Iterators
    /// @{
    const_iterator begin() const
    {
        if (!is_object()) [[unlikely]] {
            return make_iterator(rapid_const_iter{});
        }
        return make_iterator(val_->MemberBegin());
    }

    const_iterator end() const
    {
        if (!is_object()) [[unlikely]] {
            return make_iterator(rapid_const_iter{});
        }
        return make_iterator(val_->MemberEnd());
    }

    array_const_iterator array_begin() const
    {
        return make_iterator(val_->Begin());
    }

    array_const_iterator array_end() const
    {
        return make_iterator(val_->End());
    }
    /// @}

    /// @name operator[] overloads
    /// @{
    inline const_json operator[](std::string_view sv) const noexcept
    {
        ASSERT(is_object());
        return const_json((*val_)[sv.data()], doc_);
    }

    inline const_json operator[](std::size_t i) const noexcept
    {
        ASSERT(is_array());
        return const_json((*val_)[i], doc_);
    }
    /// @}

    /// @name Accessors
    /// @{
    inline bool contains(std::string_view key) const noexcept
    {
        return (is_object() && val_->HasMember(key.data()));
    }

    inline bool is_null() const noexcept
    {
        return val_->IsNull() || (is_object() && val_->ObjectEmpty());
    }

    inline bool empty() const noexcept
    {
        return is_null() || (is_array() && val_->Empty());
    }

    inline bool is_array() const noexcept
    {
        return val_->IsArray();
    }

    inline bool is_string() const noexcept
    {
        return val_->IsString();
    }

    inline bool is_bool() const noexcept
    {
        return val_->IsBool();
    }

    inline bool is_int32() const noexcept
    {
        return val_->IsInt();
    }

    inline bool is_uint32() const noexcept
    {
        return val_->IsUint();
    }

    inline bool is_int64() const noexcept
    {
        return val_->IsInt64();
    }

    inline bool is_uint64() const noexcept
    {
        return val_->IsUint64();
    }

    inline bool is_number_integer() const noexcept
    {
        return val_->IsInt() || val_->IsInt64();
    }

    inline bool is_number_unsigned() const noexcept
    {
        return val_->IsUint() || val_->IsUint64();
    }

    inline bool is_number_float() const noexcept
    {
        return val_->IsDouble();
    }

    inline bool is_float() const noexcept
    {
        return val_->IsFloat();
    }

    inline bool is_double() const noexcept
    {
        return val_->IsDouble();
    }

    inline bool is_number() const noexcept
    {
        return val_->IsNumber();
    }

    inline bool is_object() const noexcept
    {
        return val_->IsObject();
    }

    inline auto size() const noexcept
    {
        ASSERT(is_array());
        return val_->Size();
    }

    inline bool as_bool() const noexcept
    {
        return val_->GetBool();
    }

    inline int64_t as_int64() const noexcept
    {
        return val_->GetInt64();
    }

    inline uint64_t as_uint64() const noexcept
    {
        return val_->GetUint64();
    }

    inline float as_float() const noexcept
    {
        return val_->GetFloat();
    }

    inline double as_double() const noexcept
    {
        return val_->GetDouble();
    }

    inline std::string_view as_string() const noexcept
    {
        return {val_->GetString(), val_->GetStringLength()};
    }

    template <typename T>
    requires (!impl::is_vector<T> && !impl::is_std_array<T>)
    inline T get() const noexcept
    {
        if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
            return T(as_string());
        } else if constexpr (std::is_same_v<T, const_json>) {
            return const_json(*val_, doc_);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return static_cast<uint16_t>(val_->GetUint());
        } else if constexpr (std::is_same_v<T, base::half>) {
            return static_cast<base::half>(val_->GetFloat());
        } else {
            return val_->Get<T>();
        }
    }

    template <typename T>
    requires (impl::is_vector<T>)
    inline T get() const noexcept
    {
        using U = typename T::value_type;
        T result;
        if (!val_->IsArray()) [[unlikely]] {
            return result;
        }

        result.reserve(val_->Size());
        for (const auto& v : val_->GetArray()) {
            result.push_back(const_json(v, doc_).template get<U>());
        }
        return result;
    }

    template <typename T>
    requires (impl::is_std_array<T>)
    inline T get() const noexcept
    {
        using U = typename T::value_type;
        constexpr auto N = std::tuple_size<T>::value;

        T result{};
        if (!val_->IsArray() || val_->Size() != N) [[unlikely]] {
            return result;
        }

        for (auto i = 0; i < N; ++i) {
            result[i] = const_json((*val_)[i], doc_).template get<U>();
        }
        return result;
    }

    template <typename T>
    inline void get_to(T& t) const noexcept
    {
        t = get<T>();
    }

    inline auto at(std::string_view sv) const
    {
        if (!is_object() || !contains(sv)) {
            throw icm::exception("Key not found");
        }
        return operator[](sv);
    }

    inline auto at(std::size_t i) const
    {
        if (!is_array() || i >= size()) {
            throw icm::exception("Index out of range");
        }
        return operator[](i);
    }

    inline const_iterator find(std::string_view k) const noexcept
    {
        if (!is_object()) [[unlikely]] {
            return make_iterator(rapid_const_iter{});
        }
        return make_iterator(val_->FindMember(k.data()));
    }

    template <typename T>
    inline T value(std::string_view key, T default_value) const noexcept
    {
        if (auto it = find(key); it != end()) {
            return it->second.get<T>();
        }
        return default_value;
    }

    inline auto back() const noexcept
    {
        return operator[](size() - 1);
    }

    inline auto front() const noexcept
    {
        return operator[](0);
    }

    inline std::string dump() const noexcept
    {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        val_->Accept(writer);
        return buffer.GetString();
    }
    /// @}

    /// @name Comparison operators
    /// @{
    inline std::strong_ordering operator<=>(const const_json& rhs) const noexcept
    {
        if (*this == rhs) {
            return std::strong_ordering::equal;
        }
        if (*val_ < *rhs.val_) {
            return std::strong_ordering::less;
        }
        return std::strong_ordering::greater;
    }

    template <typename T>
    requires (base::arithmetic<T> || std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>)
    inline std::strong_ordering operator<=>(const T& other) const noexcept
    {
        return (*this <=> const_json(other));
    }

    inline bool operator==(const const_json& rhs) const noexcept
    {
        if ((is_bool() && rhs.is_number()) || (is_number() && rhs.is_bool())) {
            return impl::bool_number_equal(*val_, *rhs.val_);
        }
        return (*val_ == *rhs.val_);
    }

    inline bool operator==(std::string_view sv) const noexcept
    {
        return is_string() && as_string() == sv;
    }

    template <typename T>
    requires (base::arithmetic<T> || std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>)
    friend bool operator==(T lhs, const const_json& rhs) noexcept
    {
        return (rhs == const_json(lhs));
    }
    /// @}

    friend std::ostream& operator<<(std::ostream& os, const const_json& j)
    {
        os << j.dump();
        return os;
    }

private:
    template <typename T>
    requires (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view> || std::is_convertible_v<T, std::string_view>)
    inline void parse_internal(const T& v) noexcept
    {
        doc_->Parse(std::string_view(v).data(), std::string_view(v).size());
    }

    template <class CharT>
    inline void parse_internal(std::basic_istream<CharT>& s) noexcept
    {
        rapidjson::IStreamWrapper isw(s);
        doc_->ParseStream(isw);
    }

private:
    std::shared_ptr<rapidjson::Document> doc_;
    const rapidjson::Value* val_;
};

} // namespace icm


template <>
struct std::less<icm::const_json>
{
    inline bool operator()(const icm::const_json& lhs, const icm::const_json& rhs) const noexcept
    {
        return (lhs < rhs);
    }
};

template <>
struct std::greater<icm::const_json>
{
    inline bool operator()(const icm::const_json& lhs, const icm::const_json& rhs) const noexcept
    {
        return (lhs > rhs);
    }
};

template <>
struct std::equal_to<icm::const_json>
{
    inline bool operator()(const icm::const_json& lhs, const icm::const_json& rhs) const noexcept
    {
        return (lhs == rhs);
    }
};

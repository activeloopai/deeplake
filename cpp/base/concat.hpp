#pragma once

#include <sstream>
#include <string>
#include <utility>

namespace base {

namespace impl {

template <typename Arg>
inline void recursive_concat(std::ostream& stream, const std::string& separator, Arg&& arg)
{
  stream << arg << separator;
}

template <typename Arg, typename... Args>
inline void recursive_concat(std::ostream& stream, const std::string& separator, Arg&& arg, Args&&... args)
{
    recursive_concat(stream, separator, std::forward<Arg>(arg));
    recursive_concat(stream, separator, std::forward<Args>(args)...);
}

}

template <typename... Args>
inline std::string concat(const std::string& separator, Args&&... args)
{
    std::ostringstream ss;
    impl::recursive_concat(ss, separator, std::forward<Args>(args)...);
    return ss.str();
}

}
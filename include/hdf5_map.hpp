#pragma once
#include <map>
#include <string>

namespace vapor {



/**
 * Enables writing of std::map<std::string, U> to HDF5 as a group
 *
 */
template<typename U>
struct vapor::is_key_value_container_t<std::map<std::string, U>> : public std::true_type
{
};

} // namespace vapor

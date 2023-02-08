#pragma once

#include <functional>

namespace visnav {

// Utilities for hash functions to support std::pair (which std::hash doesn't
// yet).
//
// Some inspiration from
// https://www.variadic.xyz/2018/01/15/hashing-stdpair-and-stdtuple/
//
// On why XOR is not a good choice for hash-combining:
// https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
//
// hash_combine is taken more or less from boost

// forward declaration
template <typename T>
inline void hash_combine(std::size_t& seed, const T& val);

// default to std::hash
template <typename T>
struct visnav_hash : public std::hash<T> {};

// specialize for std::pair
template <typename S, typename T>
struct visnav_hash<std::pair<S, T>> {
  inline std::size_t operator()(const std::pair<S, T>& val) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, val.first);
    hash_combine(seed, val.second);
    return seed;
  }
};

template <typename T>
inline void hash_combine(std::size_t& seed, const T& val) {
  static_assert(sizeof(size_t) == sizeof(uint64_t),
                "hash_combine is meant for 64bit size_t");

  const size_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  visnav_hash<T> hasher;
  std::size_t hash = hasher(val);

  hash *= m;
  hash ^= hash >> r;
  hash *= m;

  seed ^= hash;
  seed *= m;

  // Completely arbitrary number, to prevent 0's from hashing to 0.
  seed += 0xe6546b64;
}

}  // namespace visnav

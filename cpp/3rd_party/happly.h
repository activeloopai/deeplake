#pragma once

/* A header-only implementation of the .ply file format.
 * https://github.com/nmwsharp/happly
 * By Nicholas Sharp - nsharp@cs.cmu.edu
 *
 * Version 2, July 20, 2019
 */

/*
MIT License

Copyright (c) 2018 Nick Sharp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


// clang-format off
/*

 === Changelog ===

  Significant changes to the file recorded here.

  - Version 5 (Aug 22, 2020)      Minor: skip blank lines before properties in ASCII files
  - Version 4 (Sep 11, 2019)      Change internal list format to be flat. Other small perf fixes and cleanup.
  - Version 3 (Aug 1, 2019)       Add support for big endian and obj_info
  - Version 2 (July 20, 2019)     Catch exceptions by const reference.
  - Version 1 (undated)           Initial version. Unnamed changes before version numbering.

*/
// clang-format on

#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <climits>

// General namespace wrapping all Happly things.
namespace happly {

// Enum specifying binary or ASCII filetypes. Binary can be little-endian
// (default) or big endian.
enum class DataFormat { ASCII, Binary, BinaryBigEndian };

// Type name strings
// clang-format off
template <typename T> std::string typeName()                { return "unknown"; }
template<> inline std::string typeName<int8_t>()            { return "char";    }
template<> inline std::string typeName<uint8_t>()           { return "uchar";   }
template<> inline std::string typeName<int16_t>()           { return "short";   }
template<> inline std::string typeName<uint16_t>()          { return "ushort";  }
template<> inline std::string typeName<int32_t>()           { return "int";     }
template<> inline std::string typeName<uint32_t>()          { return "uint";    }
template<> inline std::string typeName<float>()             { return "float";   }
template<> inline std::string typeName<double>()            { return "double";  }

// Template hackery that makes getProperty<T>() and friends pretty while automatically picking up smaller types
namespace {

// A pointer for the equivalent/smaller equivalent of a type (eg. when a double is requested a float works too, etc)
// long int is intentionally absent to avoid platform confusion
template <class T> struct TypeChain                 { bool hasChildType = false;   typedef T            type; };
template <> struct TypeChain<int64_t>               { bool hasChildType = true;    typedef int32_t      type; };
template <> struct TypeChain<int32_t>               { bool hasChildType = true;    typedef int16_t      type; };
template <> struct TypeChain<int16_t>               { bool hasChildType = true;    typedef int8_t       type; };
template <> struct TypeChain<uint64_t>              { bool hasChildType = true;    typedef uint32_t     type; };
template <> struct TypeChain<uint32_t>              { bool hasChildType = true;    typedef uint16_t     type; };
template <> struct TypeChain<uint16_t>              { bool hasChildType = true;    typedef uint8_t      type; };
template <> struct TypeChain<double>                { bool hasChildType = true;    typedef float        type; };

template <class T> struct CanonicalName                     { typedef T         type; };
template <> struct CanonicalName<char>                      { typedef int8_t    type; };
template <> struct CanonicalName<unsigned char>             { typedef uint8_t   type; };
template <> struct CanonicalName<size_t>                    { typedef std::conditional<std::is_same<std::make_signed<size_t>::type, int>::value, uint32_t, uint64_t>::type type; };

// Used to change behavior of >> for 8bit ints, which does not do what we want.
template <class T> struct SerializeType                 { typedef T         type; };
template <> struct SerializeType<uint8_t>               { typedef int32_t   type; };
template <> struct SerializeType< int8_t>               { typedef int32_t   type; };

// Give address only if types are same (used below when conditionally copying data)
// last int/char arg is to resolve ambiguous overloads, just always pass 0 and the int version will be preferred
template <typename S, typename T>
S* addressIfSame(T&, char) {
  throw std::runtime_error("tried to take address for types that are not same");
  return nullptr;}
template <typename S>
S* addressIfSame(S& t, int) {return &t;}

// clang-format on
} // namespace

/**
 * @brief A generic property, which is associated with some element. Can be plain Property or a ListProperty, of some
 * type.  Generally, the user should not need to interact with these directly, but they are exposed in case someone
 * wants to get clever.
 */
class Property {

public:
  /**
   * @brief Create a new Property with the given name.
   *
   * @param name_
   */
  Property(const std::string& name_) : name(name_){};
  virtual ~Property(){};

  std::string name;

  /**
   * @brief Reserve memory.
   *
   * @param capacity Expected number of elements.
   */
  virtual void reserve(size_t capacity) = 0;

  /**
   * @brief (ASCII reading) Parse out the next value of this property from a list of tokens.
   *
   * @param tokens The list of property tokens for the element.
   * @param currEntry Index in to tokens, updated after this property is read.
   */
  virtual void parseNext(const std::vector<std::string>& tokens, size_t& currEntry) = 0;

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNext(std::istream& stream) = 0;

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNextBigEndian(std::istream& stream) = 0;

  /**
   * @brief (reading) Write a header entry for this property.
   *
   * @param outStream Stream to write to.
   */
  virtual void writeHeader(std::ostream& outStream) = 0;

  /**
   * @brief (ASCII writing) write this property for some element to a stream in plaintext
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataASCII(std::ostream& outStream, size_t iElement) = 0;

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinary(std::ostream& outStream, size_t iElement) = 0;

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinaryBigEndian(std::ostream& outStream, size_t iElement) = 0;

  /**
   * @brief Number of element entries for this property
   *
   * @return
   */
  virtual size_t size() = 0;

  /**
   * @brief A string naming the type of the property
   *
   * @return
   */
  virtual std::string propertyTypeName() = 0;
};

namespace {

/**
 * Check if the platform is little endian.
 * (not foolproof, but will work on most platforms)
 *
 * @return true if little endian
 */
bool isLittleEndian() {
  int32_t oneVal = 0x1;
  char* numPtr = (char*)&oneVal;
  return (numPtr[0] == 1);
}

/**
 * Swap endianness.
 *
 * @param value Value to swap.
 *
 * @return Swapped value.
 */
template <typename T>
T swapEndian(T val) {
  char* bytes = reinterpret_cast<char*>(&val);
  for (unsigned int i = 0; i < sizeof(val) / 2; i++) {
    std::swap(bytes[sizeof(val) - 1 - i], bytes[i]);
  }
  return val;
}

// The following specializations for single-byte types are used to avoid compiler warnings.
template <> int8_t swapEndian<int8_t>(int8_t val) { return val; }
template <> uint8_t swapEndian<uint8_t>(uint8_t val) { return val; }


// Unpack flattened list from the convention used in TypedListProperty
template <typename T>
std::vector<std::vector<T>> unflattenList(const std::vector<T>& flatList, const std::vector<size_t> flatListStarts) {
  size_t outerCount = flatListStarts.size() - 1;

  // Put the output here
  std::vector<std::vector<T>> outLists(outerCount);

  if (outerCount == 0) {
    return outLists; // quick out for empty
  }

  // Copy each sublist
  for (size_t iOuter = 0; iOuter < outerCount; iOuter++) {
    size_t iFlatStart = flatListStarts[iOuter];
    size_t iFlatEnd = flatListStarts[iOuter + 1];
    outLists[iOuter].insert(outLists[iOuter].begin(), flatList.begin() + iFlatStart, flatList.begin() + iFlatEnd);
  }

  return outLists;
}


}; // namespace


/**
 * @brief A property which takes a single value (not a list).
 */
template <class T>
class TypedProperty : public Property {

public:
  /**
   * @brief Create a new Property with the given name.
   *
   * @param name_
   */
  TypedProperty(const std::string& name_) : Property(name_) {
    if (typeName<T>() == "unknown") {
      // TODO should really be a compile-time error
      throw std::runtime_error("Attempted property type does not match any type defined by the .ply format.");
    }
  };

  /**
   * @brief Create a new property and initialize with data.
   *
   * @param name_
   * @param data_
   */
  TypedProperty(const std::string& name_, const std::vector<T>& data_) : Property(name_), data(data_) {
    if (typeName<T>() == "unknown") {
      throw std::runtime_error("Attempted property type does not match any type defined by the .ply format.");
    }
  };

  virtual ~TypedProperty() override{};

  /**
   * @brief Reserve memory.
   *
   * @param capacity Expected number of elements.
   */
  virtual void reserve(size_t capacity) override { data.reserve(capacity); }

  /**
   * @brief (ASCII reading) Parse out the next value of this property from a list of tokens.
   *
   * @param tokens The list of property tokens for the element.
   * @param currEntry Index in to tokens, updated after this property is read.
   */
  virtual void parseNext(const std::vector<std::string>& tokens, size_t& currEntry) override {
    data.emplace_back();
    std::istringstream iss(tokens[currEntry]);
    typename SerializeType<T>::type tmp; // usually the same type as T
    iss >> tmp;
    data.back() = tmp;
    currEntry++;
  };

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNext(std::istream& stream) override {
    data.emplace_back();
    stream.read((char*)&data.back(), sizeof(T));
  }

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNextBigEndian(std::istream& stream) override {
    data.emplace_back();
    stream.read((char*)&data.back(), sizeof(T));
    data.back() = swapEndian(data.back());
  }

  /**
   * @brief (reading) Write a header entry for this property.
   *
   * @param outStream Stream to write to.
   */
  virtual void writeHeader(std::ostream& outStream) override {
    outStream << "property " << typeName<T>() << " " << name << "\n";
  }

  /**
   * @brief (ASCII writing) write this property for some element to a stream in plaintext
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataASCII(std::ostream& outStream, size_t iElement) override {
    outStream.precision(std::numeric_limits<T>::max_digits10);
    outStream << static_cast<typename SerializeType<T>::type>(data[iElement]); // case is usually a no-op
  }

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinary(std::ostream& outStream, size_t iElement) override {
    outStream.write((char*)&data[iElement], sizeof(T));
  }

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinaryBigEndian(std::ostream& outStream, size_t iElement) override {
    auto value = swapEndian(data[iElement]);
    outStream.write((char*)&value, sizeof(T));
  }

  /**
   * @brief Number of element entries for this property
   *
   * @return
   */
  virtual size_t size() override { return data.size(); }


  /**
   * @brief A string naming the type of the property
   *
   * @return
   */
  virtual std::string propertyTypeName() override { return typeName<T>(); }

  /**
   * @brief The actual data contained in the property
   */
  std::vector<T> data;
};


/**
 * @brief A property which is a list of value (eg, 3 doubles). Note that lists are always variable length per-element.
 */
template <class T>
class TypedListProperty : public Property {

public:
  /**
   * @brief Create a new Property with the given name.
   *
   * @param name_
   */
  TypedListProperty(const std::string& name_, int listCountBytes_) : Property(name_), listCountBytes(listCountBytes_) {
    if (typeName<T>() == "unknown") {
      throw std::runtime_error("Attempted property type does not match any type defined by the .ply format.");
    }

    flattenedIndexStart.push_back(0);
  };

  /**
   * @brief Create a new property and initialize with data
   *
   * @param name_
   * @param data_
   */
  TypedListProperty(const std::string& name_, const std::vector<std::vector<T>>& data_) : Property(name_) {
    if (typeName<T>() == "unknown") {
      throw std::runtime_error("Attempted property type does not match any type defined by the .ply format.");
    }

    // Populate list with data
    flattenedIndexStart.push_back(0);
    for (const std::vector<T>& vec : data_) {
      for (const T& val : vec) {
        flattenedData.emplace_back(val);
      }
      flattenedIndexStart.push_back(flattenedData.size());
    }
  };

  virtual ~TypedListProperty() override{};

  /**
   * @brief Reserve memory.
   *
   * @param capacity Expected number of elements.
   */
  virtual void reserve(size_t capacity) override {
    flattenedData.reserve(3 * capacity); // optimize for triangle meshes
    flattenedIndexStart.reserve(capacity + 1);
  }

  /**
   * @brief (ASCII reading) Parse out the next value of this property from a list of tokens.
   *
   * @param tokens The list of property tokens for the element.
   * @param currEntry Index in to tokens, updated after this property is read.
   */
  virtual void parseNext(const std::vector<std::string>& tokens, size_t& currEntry) override {

    std::istringstream iss(tokens[currEntry]);
    size_t count;
    iss >> count;
    currEntry++;

    size_t currSize = flattenedData.size();
    size_t afterSize = currSize + count;
    flattenedData.resize(afterSize);
    for (size_t iFlat = currSize; iFlat < afterSize; iFlat++) {
      std::istringstream iss(tokens[currEntry]);
      typename SerializeType<T>::type tmp; // usually the same type as T
      iss >> tmp;
      flattenedData[iFlat] = tmp;
      currEntry++;
    }
    flattenedIndexStart.emplace_back(afterSize);
  }

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNext(std::istream& stream) override {

    // Read the size of the list
    size_t count = 0;
    stream.read(((char*)&count), listCountBytes);

    // Read list elements
    size_t currSize = flattenedData.size();
    size_t afterSize = currSize + count;
    flattenedData.resize(afterSize);
    if (count > 0) {
      stream.read((char*)&flattenedData[currSize], count * sizeof(T));
    }
    flattenedIndexStart.emplace_back(afterSize);
  }

  /**
   * @brief (binary reading) Copy the next value of this property from a stream of bits.
   *
   * @param stream Stream to read from.
   */
  virtual void readNextBigEndian(std::istream& stream) override {

    // Read the size of the list
    size_t count = 0;
    stream.read(((char*)&count), listCountBytes);
    if (listCountBytes == 8) {
      count = (size_t)swapEndian((uint64_t)count);
    } else if (listCountBytes == 4) {
      count = (size_t)swapEndian((uint32_t)count);
    } else if (listCountBytes == 2) {
      count = (size_t)swapEndian((uint16_t)count);
    }

    // Read list elements
    size_t currSize = flattenedData.size();
    size_t afterSize = currSize + count;
    flattenedData.resize(afterSize);
    if (count > 0) {
      stream.read((char*)&flattenedData[currSize], count * sizeof(T));
    }
    flattenedIndexStart.emplace_back(afterSize);

    // Swap endian order of list elements
    for (size_t iFlat = currSize; iFlat < afterSize; iFlat++) {
      flattenedData[iFlat] = swapEndian(flattenedData[iFlat]);
    }
  }

  /**
   * @brief (reading) Write a header entry for this property. Note that we already use "uchar" for the list count type.
   *
   * @param outStream Stream to write to.
   */
  virtual void writeHeader(std::ostream& outStream) override {
    // NOTE: We ALWAYS use uchar as the list count output type
    outStream << "property list uchar " << typeName<T>() << " " << name << "\n";
  }

  /**
   * @brief (ASCII writing) write this property for some element to a stream in plaintext
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataASCII(std::ostream& outStream, size_t iElement) override {
    size_t dataStart = flattenedIndexStart[iElement];
    size_t dataEnd = flattenedIndexStart[iElement + 1];

    // Get the number of list elements as a uchar, and ensure the value fits
    size_t dataCount = dataEnd - dataStart;
    if (dataCount > std::numeric_limits<uint8_t>::max()) {
      throw std::runtime_error(
          "List property has an element with more entries than fit in a uchar. See note in README.");
    }

    outStream << dataCount;
    outStream.precision(std::numeric_limits<T>::max_digits10);
    for (size_t iFlat = dataStart; iFlat < dataEnd; iFlat++) {
      outStream << " " << static_cast<typename SerializeType<T>::type>(flattenedData[iFlat]); // cast is usually a no-op
    }
  }

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinary(std::ostream& outStream, size_t iElement) override {
    size_t dataStart = flattenedIndexStart[iElement];
    size_t dataEnd = flattenedIndexStart[iElement + 1];

    // Get the number of list elements as a uchar, and ensure the value fits
    size_t dataCount = dataEnd - dataStart;
    if (dataCount > std::numeric_limits<uint8_t>::max()) {
      throw std::runtime_error(
          "List property has an element with more entries than fit in a uchar. See note in README.");
    }
    uint8_t count = static_cast<uint8_t>(dataCount);

    outStream.write((char*)&count, sizeof(uint8_t));
    outStream.write((char*)&flattenedData[dataStart], count * sizeof(T));
  }

  /**
   * @brief (binary writing) copy the bits of this property for some element to a stream
   *
   * @param outStream Stream to write to.
   * @param iElement index of the element to write.
   */
  virtual void writeDataBinaryBigEndian(std::ostream& outStream, size_t iElement) override {
    size_t dataStart = flattenedIndexStart[iElement];
    size_t dataEnd = flattenedIndexStart[iElement + 1];

    // Get the number of list elements as a uchar, and ensure the value fits
    size_t dataCount = dataEnd - dataStart;
    if (dataCount > std::numeric_limits<uint8_t>::max()) {
      throw std::runtime_error(
          "List property has an element with more entries than fit in a uchar. See note in README.");
    }
    uint8_t count = static_cast<uint8_t>(dataCount);

    outStream.write((char*)&count, sizeof(uint8_t));
    for (size_t iFlat = dataStart; iFlat < dataEnd; iFlat++) {
      T value = swapEndian(flattenedData[iFlat]);
      outStream.write((char*)&value, sizeof(T));
    }
  }

  /**
   * @brief Number of element entries for this property
   *
   * @return
   */
  virtual size_t size() override { return flattenedIndexStart.size() - 1; }


  /**
   * @brief A string naming the type of the property
   *
   * @return
   */
  virtual std::string propertyTypeName() override { return typeName<T>(); }

  /**
   * @brief The (flattened) data for the property, as formed by concatenating all of the individual element lists
   * together.
   */
  std::vector<T> flattenedData;

  /**
   * @brief Indices in to flattenedData. The i'th element gives the index in to flattenedData where the element's data
   * begins. A final entry is included which is the length of flattenedData. Size is N_elem + 1.
   */
  std::vector<size_t> flattenedIndexStart;

  /**
   * @brief The number of bytes used to store the count for lists of data.
   */
  int listCountBytes = -1;
};


/**
 * @brief Helper function to construct a new property of the appropriate type.
 *
 * @param name The name of the property to construct.
 * @param typeStr A string naming the type according to the format.
 * @param isList Is this a plain property, or a list property?
 * @param listCountTypeStr If a list property, the type of the count varible.
 *
 * @return A new Property with the proper type.
 */
inline std::unique_ptr<Property> createPropertyWithType(const std::string& name, const std::string& typeStr,
                                                        bool isList, const std::string& listCountTypeStr) {

  // == Figure out how many bytes the list count field has, if this is a list type
  // Note: some files seem to use signed types here, we read the width but always parse as if unsigned
  int listCountBytes = -1;
  if (isList) {
    if (listCountTypeStr == "uchar" || listCountTypeStr == "uint8" || listCountTypeStr == "char" ||
        listCountTypeStr == "int8") {
      listCountBytes = 1;
    } else if (listCountTypeStr == "ushort" || listCountTypeStr == "uint16" || listCountTypeStr == "short" ||
               listCountTypeStr == "int16") {
      listCountBytes = 2;
    } else if (listCountTypeStr == "uint" || listCountTypeStr == "uint32" || listCountTypeStr == "int" ||
               listCountTypeStr == "int32") {
      listCountBytes = 4;
    } else {
      throw std::runtime_error("Unrecognized list count type: " + listCountTypeStr);
    }
  }

  // = Unsigned int

  // 8 bit unsigned
  if (typeStr == "uchar" || typeStr == "uint8") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<uint8_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<uint8_t>(name));
    }
  }

  // 16 bit unsigned
  else if (typeStr == "ushort" || typeStr == "uint16") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<uint16_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<uint16_t>(name));
    }
  }

  // 32 bit unsigned
  else if (typeStr == "uint" || typeStr == "uint32") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<uint32_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<uint32_t>(name));
    }
  }

  // = Signed int

  // 8 bit signed
  if (typeStr == "char" || typeStr == "int8") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<int8_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<int8_t>(name));
    }
  }

  // 16 bit signed
  else if (typeStr == "short" || typeStr == "int16") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<int16_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<int16_t>(name));
    }
  }

  // 32 bit signed
  else if (typeStr == "int" || typeStr == "int32") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<int32_t>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<int32_t>(name));
    }
  }

  // = Float

  // 32 bit float
  else if (typeStr == "float" || typeStr == "float32") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<float>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<float>(name));
    }
  }

  // 64 bit float
  else if (typeStr == "double" || typeStr == "float64") {
    if (isList) {
      return std::unique_ptr<Property>(new TypedListProperty<double>(name, listCountBytes));
    } else {
      return std::unique_ptr<Property>(new TypedProperty<double>(name));
    }
  }

  else {
    throw std::runtime_error("Data type: " + typeStr + " cannot be mapped to .ply format");
  }
}

/**
 * @brief An element (more properly an element type) in the .ply object. Tracks the name of the elemnt type (eg,
 * "vertices"), the number of elements of that type (eg, 1244), and any properties associated with that element (eg,
 * "position", "color").
 */
class Element {

public:
  /**
   * @brief Create a new element type.
   *
   * @param name_ Name of the element type (eg, "vertices")
   * @param count_ Number of instances of this element.
   */
  Element(const std::string& name_, size_t count_) : name(name_), count(count_) {}

  std::string name;
  size_t count;
  std::vector<std::unique_ptr<Property>> properties;

  /**
   * @brief Check if a property exists.
   *
   * @param target The name of the property to get.
   *
   * @return Whether the target property exists.
   */
  bool hasProperty(const std::string& target) {
    for (std::unique_ptr<Property>& prop : properties) {
      if (prop->name == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Check if a property exists with the requested type.
   *
   * @tparam T The type of the property
   * @param target The name of the property to get.
   *
   * @return Whether the target property exists.
   */
  template <class T>
  bool hasPropertyType(const std::string& target) {
    for (std::unique_ptr<Property>& prop : properties) {
      if (prop->name == target) {
        TypedProperty<T>* castedProp = dynamic_cast<TypedProperty<T>*>(prop.get());
        if (castedProp) {
          return true;
        }
        return false;
      }
    }
    return false;
  }

  /**
   * @brief A list of the names of all properties
   *
   * @return Property names
   */
  std::vector<std::string> getPropertyNames() {
    std::vector<std::string> names;
    for (std::unique_ptr<Property>& p : properties) {
      names.push_back(p->name);
    }
    return names;
  }

  /**
   * @brief Low-level method to get a pointer to a property. Users probably don't need to call this.
   *
   * @param target The name of the property to get.
   *
   * @return A (unique_ptr) pointer to the property.
   */
  std::unique_ptr<Property>& getPropertyPtr(const std::string& target) {
    for (std::unique_ptr<Property>& prop : properties) {
      if (prop->name == target) {
        return prop;
      }
    }
    throw std::runtime_error("PLY parser: element " + name + " does not have property " + target);
  }

  /**
   * @brief Add a new (plain, not list) property for this element type.
   *
   * @tparam T The type of the property
   * @param propertyName The name of the property
   * @param data The data for the property. Must have the same length as the number of elements.
   */
  template <class T>
  void addProperty(const std::string& propertyName, const std::vector<T>& data) {

    if (data.size() != count) {
      throw std::runtime_error("PLY write: new property " + propertyName + " has size which does not match element");
    }

    // If there is already some property with this name, remove it
    for (size_t i = 0; i < properties.size(); i++) {
      if (properties[i]->name == propertyName) {
        properties.erase(properties.begin() + i);
        i--;
      }
    }

    // Copy to canonical type. Often a no-op, but takes care of standardizing widths across platforms.
    std::vector<typename CanonicalName<T>::type> canonicalVec(data.begin(), data.end());

    properties.push_back(
        std::unique_ptr<Property>(new TypedProperty<typename CanonicalName<T>::type>(propertyName, canonicalVec)));
  }

  /**
   * @brief Add a new list property for this element type.
   *
   * @tparam T The type of the property (eg, "double" for a list of doubles)
   * @param propertyName The name of the property
   * @param data The data for the property. Outer vector must have the same length as the number of elements.
   */
  template <class T>
  void addListProperty(const std::string& propertyName, const std::vector<std::vector<T>>& data) {

    if (data.size() != count) {
      throw std::runtime_error("PLY write: new property " + propertyName + " has size which does not match element");
    }

    // If there is already some property with this name, remove it
    for (size_t i = 0; i < properties.size(); i++) {
      if (properties[i]->name == propertyName) {
        properties.erase(properties.begin() + i);
        i--;
      }
    }

    // Copy to canonical type. Often a no-op, but takes care of standardizing widths across platforms.
    std::vector<std::vector<typename CanonicalName<T>::type>> canonicalListVec;
    for (const std::vector<T>& subList : data) {
      canonicalListVec.emplace_back(subList.begin(), subList.end());
    }

    properties.push_back(std::unique_ptr<Property>(
        new TypedListProperty<typename CanonicalName<T>::type>(propertyName, canonicalListVec)));
  }

  /**
   * @brief Get a vector of a data from a property for this element. Automatically promotes to larger types. Throws if
   * requested data is unavailable.
   *
   * @tparam T The type of data requested
   * @param propertyName The name of the property to get.
   *
   * @return The data.
   */
  template <class T>
  std::vector<T> getProperty(const std::string& propertyName) {

    // Find the property
    std::unique_ptr<Property>& prop = getPropertyPtr(propertyName);

    // Get a copy of the data with auto-promoting type magic
    return getDataFromPropertyRecursive<T, T>(prop.get());
  }

  /**
   * @brief Get a vector of a data from a property for this element. Unlike getProperty(), only returns if the ply
   * record contains a type that matches T exactly. Throws if * requested data is unavailable.
   *
   * @tparam T The type of data requested
   * @param propertyName The name of the property to get.
   *
   * @return The data.
   */
  template <class T>
  std::vector<T> getPropertyType(const std::string& propertyName) {

    // Find the property
    std::unique_ptr<Property>& prop = getPropertyPtr(propertyName);
    TypedProperty<T>* castedProp = dynamic_cast<TypedProperty<T>*>(prop.get());
    if (castedProp) {
      return castedProp->data;
    }

    // No match, failure
    throw std::runtime_error("PLY parser: property " + prop->name + " is not of type type " + typeName<T>() +
                             ". Has type " + prop->propertyTypeName());
  }

  /**
   * @brief Get a vector of lists of data from a property for this element. Automatically promotes to larger types.
   * Throws if requested data is unavailable.
   *
   * @tparam T The type of data requested
   * @param propertyName The name of the property to get.
   *
   * @return The data.
   */
  template <class T>
  std::vector<std::vector<T>> getListProperty(const std::string& propertyName) {

    // Find the property
    std::unique_ptr<Property>& prop = getPropertyPtr(propertyName);

    // Get a copy of the data with auto-promoting type magic
    return getDataFromListPropertyRecursive<T, T>(prop.get());
  }

  /**
   * @brief Get a vector of a data from a property for this element. Unlike getProperty(), only returns if the ply
   * record contains a type that matches T exactly. Throws if * requested data is unavailable.
   *
   * @tparam T The type of data requested
   * @param propertyName The name of the property to get.
   *
   * @return The data.
   */
  template <class T>
  std::vector<std::vector<T>> getListPropertyType(const std::string& propertyName) {

    // Find the property
    std::unique_ptr<Property>& prop = getPropertyPtr(propertyName);
    TypedListProperty<T>* castedProp = dynamic_cast<TypedListProperty<T>*>(prop.get());
    if (castedProp) {
      return unflattenList(castedProp->flattenedData, castedProp->flattenedIndexStart);
    }

    // No match, failure
    throw std::runtime_error("PLY parser: list property " + prop->name + " is not of type " + typeName<T>() +
                             ". Has type " + prop->propertyTypeName());
  }


  /**
   * @brief Get a vector of lists of data from a property for this element. Automatically promotes to larger types.
   * Unlike getListProperty(), this method will additionally convert between types of different sign (eg, requesting and
   * int32 would get data from a uint32); doing so naively converts between signed and unsigned types. This is typically
   * useful for data representing indices, which might be stored as signed or unsigned numbers.
   *
   * @tparam T The type of data requested
   * @param propertyName The name of the property to get.
   *
   * @return The data.
   */
  template <class T>
  std::vector<std::vector<T>> getListPropertyAnySign(const std::string& propertyName) {

    // Find the property
    std::unique_ptr<Property>& prop = getPropertyPtr(propertyName);

    // Get a copy of the data with auto-promoting type magic
    try {
      // First, try the usual approach, looking for a version of the property with the same signed-ness and possibly
      // smaller size
      return getDataFromListPropertyRecursive<T, T>(prop.get());
    } catch (const std::runtime_error& orig_e) {

      // If the usual approach fails, look for a version with opposite signed-ness
      try {

        // This type has the oppopsite signeness as the input type
        typedef typename CanonicalName<T>::type Tcan;
        typedef typename std::conditional<std::is_signed<Tcan>::value, typename std::make_unsigned<Tcan>::type,
                                          typename std::make_signed<Tcan>::type>::type OppsignType;

        return getDataFromListPropertyRecursive<T, OppsignType>(prop.get());

      } catch (const std::runtime_error&) {
        throw orig_e;
      }

      throw orig_e;
    }
  }


  /**
   * @brief Performs sanity checks on the element, throwing if any fail.
   */
  void validate() {

    // Make sure no properties have duplicate names, and no names have whitespace
    for (size_t iP = 0; iP < properties.size(); iP++) {
      for (char c : properties[iP]->name) {
        if (std::isspace(c)) {
          throw std::runtime_error("Ply validate: illegal whitespace in name " + properties[iP]->name);
        }
      }
      for (size_t jP = iP + 1; jP < properties.size(); jP++) {
        if (properties[iP]->name == properties[jP]->name) {
          throw std::runtime_error("Ply validate: multiple properties with name " + properties[iP]->name);
        }
      }
    }

    // Make sure all properties have right length
    for (size_t iP = 0; iP < properties.size(); iP++) {
      if (properties[iP]->size() != count) {
        throw std::runtime_error("Ply validate: property has wrong size. " + properties[iP]->name +
                                 " does not match element size.");
      }
    }
  }

  /**
   * @brief Writes out this element's information to the file header.
   *
   * @param outStream The stream to use.
   */
  void writeHeader(std::ostream& outStream) {

    outStream << "element " << name << " " << count << "\n";

    for (std::unique_ptr<Property>& p : properties) {
      p->writeHeader(outStream);
    }
  }

  /**
   * @brief (ASCII writing) Writes out all of the data for every element of this element type to the stream, including
   * all contained properties.
   *
   * @param outStream The stream to write to.
   */
  void writeDataASCII(std::ostream& outStream) {
    // Question: what is the proper output for an element with no properties? Here, we write a blank line, so there is
    // one line per element no matter what.
    for (size_t iE = 0; iE < count; iE++) {
      for (size_t iP = 0; iP < properties.size(); iP++) {
        properties[iP]->writeDataASCII(outStream, iE);
        if (iP < properties.size() - 1) {
          outStream << " ";
        }
      }
      outStream << "\n";
    }
  }


  /**
   * @brief (binary writing) Writes out all of the data for every element of this element type to the stream, including
   * all contained properties.
   *
   * @param outStream The stream to write to.
   */
  void writeDataBinary(std::ostream& outStream) {
    for (size_t iE = 0; iE < count; iE++) {
      for (size_t iP = 0; iP < properties.size(); iP++) {
        properties[iP]->writeDataBinary(outStream, iE);
      }
    }
  }


  /**
   * @brief (binary writing) Writes out all of the data for every element of this element type to the stream, including
   * all contained properties.
   *
   * @param outStream The stream to write to.
   */
  void writeDataBinaryBigEndian(std::ostream& outStream) {
    for (size_t iE = 0; iE < count; iE++) {
      for (size_t iP = 0; iP < properties.size(); iP++) {
        properties[iP]->writeDataBinaryBigEndian(outStream, iE);
      }
    }
  }


  /**
   * @brief Helper function which does the hard work to implement type promotion for data getters. Throws if type
   * conversion fails.
   *
   * @tparam D The desired output type
   * @tparam T The current attempt for the actual type of the property
   * @param prop The property to get (does not delete nor share pointer)
   *
   * @return The data, with the requested type
   */
  template <class D, class T>
  std::vector<D> getDataFromPropertyRecursive(Property* prop) {

    typedef typename CanonicalName<T>::type Tcan;

    { // Try to return data of type D from a property of type T
      TypedProperty<Tcan>* castedProp = dynamic_cast<TypedProperty<Tcan>*>(prop);
      if (castedProp) {
        // Succeeded, return a buffer of the data (copy while converting type)
        std::vector<D> castedVec;
        castedVec.reserve(castedProp->data.size());
        for (Tcan& v : castedProp->data) {
          castedVec.push_back(static_cast<D>(v));
        }
        return castedVec;
      }
    }

    TypeChain<Tcan> chainType;
    if (chainType.hasChildType) {
      return getDataFromPropertyRecursive<D, typename TypeChain<Tcan>::type>(prop);
    } else {
      // No smaller type to try, failure
      throw std::runtime_error("PLY parser: property " + prop->name + " cannot be coerced to requested type " +
                               typeName<D>() + ". Has type " + prop->propertyTypeName());
    }
  }


  /**
   * @brief Helper function which does the hard work to implement type promotion for list data getters. Throws if type
   * conversion fails.
   *
   * @tparam D The desired output type
   * @tparam T The current attempt for the actual type of the property
   * @param prop The property to get (does not delete nor share pointer)
   *
   * @return The data, with the requested type
   */
  template <class D, class T>
  std::vector<std::vector<D>> getDataFromListPropertyRecursive(Property* prop) {
    typedef typename CanonicalName<T>::type Tcan;

    TypedListProperty<Tcan>* castedProp = dynamic_cast<TypedListProperty<Tcan>*>(prop);
    if (castedProp) {
      // Succeeded, return a buffer of the data (copy while converting type)

      // Convert to flat buffer of new type
      std::vector<D>* castedFlatVec = nullptr;
      std::vector<D> castedFlatVecCopy; // we _might_ make a copy here, depending on is_same below

      if (std::is_same<std::vector<D>, std::vector<Tcan>>::value) {
        // just use the array we already have
        castedFlatVec = addressIfSame<std::vector<D>>(castedProp->flattenedData, 0 /* dummy arg to disambiguate */);
      } else {
        // make a copy
        castedFlatVecCopy.reserve(castedProp->flattenedData.size());
        for (Tcan& v : castedProp->flattenedData) {
          castedFlatVecCopy.push_back(static_cast<D>(v));
        }
        castedFlatVec = &castedFlatVecCopy;
      }

      // Unflatten and return
      return unflattenList(*castedFlatVec, castedProp->flattenedIndexStart);
    }

    TypeChain<Tcan> chainType;
    if (chainType.hasChildType) {
      return getDataFromListPropertyRecursive<D, typename TypeChain<Tcan>::type>(prop);
    } else {
      // No smaller type to try, failure
      throw std::runtime_error("PLY parser: list property " + prop->name +
                               " cannot be coerced to requested type list " + typeName<D>() + ". Has type list " +
                               prop->propertyTypeName());
    }
  }
};


// Some string helpers
namespace {

inline std::string trimSpaces(const std::string& input) {
  size_t start = 0;
  while (start < input.size() && input[start] == ' ') start++;
  size_t end = input.size();
  while (end > start && (input[end - 1] == ' ' || input[end - 1] == '\n' || input[end - 1] == '\r')) end--;
  return input.substr(start, end - start);
}

inline std::vector<std::string> tokenSplit(const std::string& input) {
  std::vector<std::string> result;
  size_t curr = 0;
  size_t found = 0;
  while ((found = input.find_first_of(' ', curr)) != std::string::npos) {
    std::string token = input.substr(curr, found - curr);
    token = trimSpaces(token);
    if (token.size() > 0) {
      result.push_back(token);
    }
    curr = found + 1;
  }
  std::string token = input.substr(curr);
  token = trimSpaces(token);
  if (token.size() > 0) {
    result.push_back(token);
  }

  return result;
}

inline bool startsWith(const std::string& input, const std::string& query) {
  return input.compare(0, query.length(), query) == 0;
}
}; // namespace


/**
 * @brief Primary class; represents a set of data in the .ply format.
 */
class PLYData {

public:
  /**
   * @brief Create an empty PLYData object.
   */
  PLYData(){};

  /**
   * @brief Initialize a PLYData by reading from a file. Throws if any failures occur.
   *
   * @param filename The file to read from.
   * @param verbose If true, print useful info about the file to stdout
   */
  PLYData(const std::string& filename, bool verbose = false) {

    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;

    if (verbose) cout << "PLY parser: Reading ply file: " << filename << endl;

    // Open a file in binary always, in case it turns out to have binary data.
    std::ifstream inStream(filename, std::ios::binary);
    if (inStream.fail()) {
      throw std::runtime_error("PLY parser: Could not open file " + filename);
    }

    parsePLY(inStream, verbose);

    if (verbose) {
      cout << "  - Finished parsing file." << endl;
    }
  }

  /**
   * @brief Initialize a PLYData by reading from a stringstream. Throws if any failures occur.
   *
   * @param inStream The stringstream to read from.
   * @param verbose If true, print useful info about the file to stdout
   */
  PLYData(std::istream& inStream, bool verbose = false) {

    using std::cout;
    using std::endl;

    if (verbose) cout << "PLY parser: Reading ply file from stream" << endl;

    parsePLY(inStream, verbose);

    if (verbose) {
      cout << "  - Finished parsing stream." << endl;
    }
  }

  /**
   * @brief Perform sanity checks on the file, throwing if any fail.
   */
  void validate() {

    for (size_t iE = 0; iE < elements.size(); iE++) {
      for (char c : elements[iE].name) {
        if (std::isspace(c)) {
          throw std::runtime_error("Ply validate: illegal whitespace in element name " + elements[iE].name);
        }
      }
      for (size_t jE = iE + 1; jE < elements.size(); jE++) {
        if (elements[iE].name == elements[jE].name) {
          throw std::runtime_error("Ply validate: duplcate element name " + elements[iE].name);
        }
      }
    }

    // Do a quick validation sanity check
    for (Element& e : elements) {
      e.validate();
    }
  }

  /**
   * @brief Write this data to a .ply file.
   *
   * @param filename The file to write to.
   * @param format The format to use (binary or ascii?)
   */
  void write(const std::string& filename, DataFormat format = DataFormat::ASCII) {
    outputDataFormat = format;

    validate();

    // Open stream for writing
    std::ofstream outStream(filename, std::ios::out | std::ios::binary);
    if (!outStream.good()) {
      throw std::runtime_error("Ply writer: Could not open output file " + filename + " for writing");
    }

    writePLY(outStream);
  }

  /**
   * @brief Write this data to an output stream
   *
   * @param outStream The output stream to write to.
   * @param format The format to use (binary or ascii?)
   */
  void write(std::ostream& outStream, DataFormat format = DataFormat::ASCII) {
    outputDataFormat = format;

    validate();

    writePLY(outStream);
  }

  /**
   * @brief Get an element type by name ("vertices")
   *
   * @param target The name of the element type to get
   *
   * @return A reference to the element type.
   */
  Element& getElement(const std::string& target) {
    for (Element& e : elements) {
      if (e.name == target) return e;
    }
    throw std::runtime_error("PLY parser: no element with name: " + target);
  }


  /**
   * @brief Check if an element type exists
   *
   * @param target The name to check for.
   *
   * @return True if exists.
   */
  bool hasElement(const std::string& target) {
    for (Element& e : elements) {
      if (e.name == target) return true;
    }
    return false;
  }


  /**
   * @brief A list of the names of all elements
   *
   * @return Element names
   */
  std::vector<std::string> getElementNames() {
    std::vector<std::string> names;
    for (Element& e : elements) {
      names.push_back(e.name);
    }
    return names;
  }


  /**
   * @brief Add a new element type to the object
   *
   * @param name The name of the new element type ("vertices").
   * @param count The number of elements of this type.
   */
  void addElement(const std::string& name, size_t count) { elements.emplace_back(name, count); }

  // === Common-case helpers


  /**
   * @brief Common-case helper get mesh vertex positions
   *
   * @param vertexElementName The element name to use (default: "vertex")
   *
   * @return A vector of vertex positions.
   */
  std::vector<std::array<double, 3>> getVertexPositions(const std::string& vertexElementName = "vertex") {

    std::vector<double> xPos = getElement(vertexElementName).getProperty<double>("x");
    std::vector<double> yPos = getElement(vertexElementName).getProperty<double>("y");
    std::vector<double> zPos = getElement(vertexElementName).getProperty<double>("z");

    std::vector<std::array<double, 3>> result(xPos.size());
    for (size_t i = 0; i < result.size(); i++) {
      result[i][0] = xPos[i];
      result[i][1] = yPos[i];
      result[i][2] = zPos[i];
    }

    return result;
  }

  /**
   * @brief Common-case helper get mesh vertex colors
   *
   * @param vertexElementName The element name to use (default: "vertex")
   *
   * @return A vector of vertex colors (unsigned chars [0,255]).
   */
  std::vector<std::array<unsigned char, 3>> getVertexColors(const std::string& vertexElementName = "vertex") {

    std::vector<unsigned char> r = getElement(vertexElementName).getProperty<unsigned char>("red");
    std::vector<unsigned char> g = getElement(vertexElementName).getProperty<unsigned char>("green");
    std::vector<unsigned char> b = getElement(vertexElementName).getProperty<unsigned char>("blue");

    std::vector<std::array<unsigned char, 3>> result(r.size());
    for (size_t i = 0; i < result.size(); i++) {
      result[i][0] = r[i];
      result[i][1] = g[i];
      result[i][2] = b[i];
    }

    return result;
  }

  /**
   * @brief Common-case helper to get face indices for a mesh. If not template type is given, size_t is used. Naively
   * converts to requested signedness, which may lead to unexpected values if an unsigned type is used and file contains
   * negative values.
   *
   * @return The indices into the vertex elements for each face. Usually 0-based, though there are no formal rules.
   */
  template <typename T = size_t>
  std::vector<std::vector<T>> getFaceIndices() {

    for (const std::string& f : std::vector<std::string>{"face"}) {
      for (const std::string& p : std::vector<std::string>{"vertex_indices", "vertex_index"}) {
        try {
          return getElement(f).getListPropertyAnySign<T>(p);
        } catch (const std::runtime_error&) {
          // that's fine
        }
      }
    }
    throw std::runtime_error("PLY parser: could not find face vertex indices attribute under any common name.");
  }


  /**
   * @brief Common-case helper set mesh vertex positons. Creates vertex element, if necessary.
   *
   * @param vertexPositions A vector of vertex positions
   */
  void addVertexPositions(std::vector<std::array<double, 3>>& vertexPositions) {

    std::string vertexName = "vertex";
    size_t N = vertexPositions.size();

    // Create the element
    if (!hasElement(vertexName)) {
      addElement(vertexName, N);
    }

    // De-interleave
    std::vector<double> xPos(N);
    std::vector<double> yPos(N);
    std::vector<double> zPos(N);
    for (size_t i = 0; i < vertexPositions.size(); i++) {
      xPos[i] = vertexPositions[i][0];
      yPos[i] = vertexPositions[i][1];
      zPos[i] = vertexPositions[i][2];
    }

    // Store
    getElement(vertexName).addProperty<double>("x", xPos);
    getElement(vertexName).addProperty<double>("y", yPos);
    getElement(vertexName).addProperty<double>("z", zPos);
  }

  /**
   * @brief Common-case helper set mesh vertex colors. Creates a vertex element, if necessary.
   *
   * @param colors A vector of vertex colors (unsigned chars [0,255]).
   */
  void addVertexColors(std::vector<std::array<unsigned char, 3>>& colors) {

    std::string vertexName = "vertex";
    size_t N = colors.size();

    // Create the element
    if (!hasElement(vertexName)) {
      addElement(vertexName, N);
    }

    // De-interleave
    std::vector<unsigned char> r(N);
    std::vector<unsigned char> g(N);
    std::vector<unsigned char> b(N);
    for (size_t i = 0; i < colors.size(); i++) {
      r[i] = colors[i][0];
      g[i] = colors[i][1];
      b[i] = colors[i][2];
    }

    // Store
    getElement(vertexName).addProperty<unsigned char>("red", r);
    getElement(vertexName).addProperty<unsigned char>("green", g);
    getElement(vertexName).addProperty<unsigned char>("blue", b);
  }

  /**
   * @brief Common-case helper set mesh vertex colors. Creates a vertex element, if necessary.
   *
   * @param colors A vector of vertex colors as floating point [0,1] values. Internally converted to [0,255] chars.
   */
  void addVertexColors(std::vector<std::array<double, 3>>& colors) {

    std::string vertexName = "vertex";
    size_t N = colors.size();

    // Create the element
    if (!hasElement(vertexName)) {
      addElement(vertexName, N);
    }

    auto toChar = [](double v) {
      if (v < 0.0) v = 0.0;
      if (v > 1.0) v = 1.0;
      return static_cast<unsigned char>(v * 255.);
    };

    // De-interleave
    std::vector<unsigned char> r(N);
    std::vector<unsigned char> g(N);
    std::vector<unsigned char> b(N);
    for (size_t i = 0; i < colors.size(); i++) {
      r[i] = toChar(colors[i][0]);
      g[i] = toChar(colors[i][1]);
      b[i] = toChar(colors[i][2]);
    }

    // Store
    getElement(vertexName).addProperty<unsigned char>("red", r);
    getElement(vertexName).addProperty<unsigned char>("green", g);
    getElement(vertexName).addProperty<unsigned char>("blue", b);
  }


  /**
   * @brief Common-case helper to set face indices. Creates a face element if needed. The input type will be casted to a
   * 32 bit integer of the same signedness.
   *
   * @param indices The indices into the vertex list around each face.
   */
  template <typename T>
  void addFaceIndices(std::vector<std::vector<T>>& indices) {

    std::string faceName = "face";
    size_t N = indices.size();

    // Create the element
    if (!hasElement(faceName)) {
      addElement(faceName, N);
    }

    // Cast to 32 bit
    typedef typename std::conditional<std::is_signed<T>::value, int32_t, uint32_t>::type IndType;
    std::vector<std::vector<IndType>> intInds;
    for (std::vector<T>& l : indices) {
      std::vector<IndType> thisInds;
      for (T& val : l) {
        IndType valConverted = static_cast<IndType>(val);
        if (valConverted != val) {
          throw std::runtime_error("Index value " + std::to_string(val) +
                                   " could not be converted to a .ply integer without loss of data. Note that .ply "
                                   "only supports 32-bit ints.");
        }
        thisInds.push_back(valConverted);
      }
      intInds.push_back(thisInds);
    }

    // Store
    getElement(faceName).addListProperty<IndType>("vertex_indices", intInds);
  }


  /**
   * @brief Comments for the file. When writing, each entry will be written as a sequential comment line.
   */
  std::vector<std::string> comments;


  /**
   * @brief obj_info comments for the file. When writing, each entry will be written as a sequential comment line.
   */
  std::vector<std::string> objInfoComments;

private:
  std::vector<Element> elements;
  const int majorVersion = 1; // I'll buy you a drink if these ever get bumped
  const int minorVersion = 0;

  DataFormat inputDataFormat = DataFormat::ASCII;  // set when reading from a file
  DataFormat outputDataFormat = DataFormat::ASCII; // option for writing files


  // === Reading ===

  /**
   * @brief Parse a PLY file from an input stream
   *
   * @param inStream
   * @param verbose
   */
  void parsePLY(std::istream& inStream, bool verbose) {

    // == Process the header
    parseHeader(inStream, verbose);


    // === Parse data from a binary file
    if (inputDataFormat == DataFormat::Binary) {
      parseBinary(inStream, verbose);
    }
    // === Parse data from an binary file
    else if (inputDataFormat == DataFormat::BinaryBigEndian) {
      parseBinaryBigEndian(inStream, verbose);
    }
    // === Parse data from an ASCII file
    else if (inputDataFormat == DataFormat::ASCII) {
      parseASCII(inStream, verbose);
    }
  }

  /**
   * @brief Read the header for a file
   *
   * @param inStream
   * @param verbose
   */
  void parseHeader(std::istream& inStream, bool verbose) {

    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;

    // First two lines are predetermined
    { // First line is magic constant
      string plyLine;
      std::getline(inStream, plyLine);
      if (trimSpaces(plyLine) != "ply") {
        throw std::runtime_error("PLY parser: File does not appear to be ply file. First line should be 'ply'");
      }
    }

    { // second line is version
      string styleLine;
      std::getline(inStream, styleLine);
      vector<string> tokens = tokenSplit(styleLine);
      if (tokens.size() != 3) throw std::runtime_error("PLY parser: bad format line");
      std::string formatStr = tokens[0];
      std::string typeStr = tokens[1];
      std::string versionStr = tokens[2];

      // "format"
      if (formatStr != "format") throw std::runtime_error("PLY parser: bad format line");

      // ascii/binary
      if (typeStr == "ascii") {
        inputDataFormat = DataFormat::ASCII;
        if (verbose) cout << "  - Type: ascii" << endl;
      } else if (typeStr == "binary_little_endian") {
        inputDataFormat = DataFormat::Binary;
        if (verbose) cout << "  - Type: binary" << endl;
      } else if (typeStr == "binary_big_endian") {
        inputDataFormat = DataFormat::BinaryBigEndian;
        if (verbose) cout << "  - Type: binary big endian" << endl;
      } else {
        throw std::runtime_error("PLY parser: bad format line");
      }

      // version
      if (versionStr != "1.0") {
        throw std::runtime_error("PLY parser: encountered file with version != 1.0. Don't know how to parse that");
      }
      if (verbose) cout << "  - Version: " << versionStr << endl;
    }

    // Consume header line by line
    while (inStream.good()) {
      string line;
      std::getline(inStream, line);

      // Parse a comment
      if (startsWith(line, "comment")) {
        string comment = line.substr(8);
        if (verbose) cout << "  - Comment: " << comment << endl;
        comments.push_back(comment);
        continue;
      }

      // Parse an obj_info comment
      if (startsWith(line, "obj_info")) {
        string infoComment = line.substr(9);
        if (verbose) cout << "  - obj_info: " << infoComment << endl;
        objInfoComments.push_back(infoComment);
        continue;
      }

      // Parse an element
      else if (startsWith(line, "element")) {
        vector<string> tokens = tokenSplit(line);
        if (tokens.size() != 3) throw std::runtime_error("PLY parser: Invalid element line");
        string name = tokens[1];
        size_t count;
        std::istringstream iss(tokens[2]);
        iss >> count;
        elements.emplace_back(name, count);
        if (verbose) cout << "  - Found element: " << name << " (count = " << count << ")" << endl;
        continue;
      }

      // Parse a property list
      else if (startsWith(line, "property list")) {
        vector<string> tokens = tokenSplit(line);
        if (tokens.size() != 5) throw std::runtime_error("PLY parser: Invalid property list line");
        if (elements.size() == 0) throw std::runtime_error("PLY parser: Found property list without previous element");
        string countType = tokens[2];
        string type = tokens[3];
        string name = tokens[4];
        elements.back().properties.push_back(createPropertyWithType(name, type, true, countType));
        if (verbose)
          cout << "    - Found list property: " << name << " (count type = " << countType << ", data type = " << type
               << ")" << endl;
        continue;
      }

      // Parse a property
      else if (startsWith(line, "property")) {
        vector<string> tokens = tokenSplit(line);
        if (tokens.size() != 3) throw std::runtime_error("PLY parser: Invalid property line");
        if (elements.size() == 0) throw std::runtime_error("PLY parser: Found property without previous element");
        string type = tokens[1];
        string name = tokens[2];
        elements.back().properties.push_back(createPropertyWithType(name, type, false, ""));
        if (verbose) cout << "    - Found property: " << name << " (type = " << type << ")" << endl;
        continue;
      }

      // Parse end of header
      else if (startsWith(line, "end_header")) {
        break;
      }

      // Error!
      else {
        throw std::runtime_error("Unrecognized header line: " + line);
      }
    }
  }

  /**
   * @brief Read the actual data for a file, in ASCII
   *
   * @param inStream
   * @param verbose
   */
  void parseASCII(std::istream& inStream, bool verbose) {

    using std::string;
    using std::vector;

    // Read all elements
    for (Element& elem : elements) {

      if (verbose) {
        std::cout << "  - Processing element: " << elem.name << std::endl;
      }

      for (size_t iP = 0; iP < elem.properties.size(); iP++) {
        elem.properties[iP]->reserve(elem.count);
      }
      for (size_t iEntry = 0; iEntry < elem.count; iEntry++) {

        string line;
        std::getline(inStream, line);

        // Some .ply files seem to include empty lines before the start of property data (though this is not specified
        // in the format description). We attempt to recover and parse such files by skipping any empty lines.
        if (!elem.properties.empty()) { // if the element has no properties, the line _should_ be blank, presumably
          while (line.empty()) { // skip lines until we hit something nonempty
            std::getline(inStream, line);
          }
        }

        vector<string> tokens = tokenSplit(line);
        size_t iTok = 0;
        for (size_t iP = 0; iP < elem.properties.size(); iP++) {
          elem.properties[iP]->parseNext(tokens, iTok);
        }
      }
    }
  }

  /**
   * @brief Read the actual data for a file, in binary.
   *
   * @param inStream
   * @param verbose
   */
  void parseBinary(std::istream& inStream, bool verbose) {

    if (!isLittleEndian()) {
      throw std::runtime_error("binary reading assumes little endian system");
    }

    using std::string;
    using std::vector;

    // Read all elements
    for (Element& elem : elements) {

      if (verbose) {
        std::cout << "  - Processing element: " << elem.name << std::endl;
      }

      for (size_t iP = 0; iP < elem.properties.size(); iP++) {
        elem.properties[iP]->reserve(elem.count);
      }
      for (size_t iEntry = 0; iEntry < elem.count; iEntry++) {
        for (size_t iP = 0; iP < elem.properties.size(); iP++) {
          elem.properties[iP]->readNext(inStream);
        }
      }
    }
  }

  /**
   * @brief Read the actual data for a file, in binary.
   *
   * @param inStream
   * @param verbose
   */
  void parseBinaryBigEndian(std::istream& inStream, bool verbose) {

    if (!isLittleEndian()) {
      throw std::runtime_error("binary reading assumes little endian system");
    }

    using std::string;
    using std::vector;

    // Read all elements
    for (Element& elem : elements) {

      if (verbose) {
        std::cout << "  - Processing element: " << elem.name << std::endl;
      }

      for (size_t iP = 0; iP < elem.properties.size(); iP++) {
        elem.properties[iP]->reserve(elem.count);
      }
      for (size_t iEntry = 0; iEntry < elem.count; iEntry++) {
        for (size_t iP = 0; iP < elem.properties.size(); iP++) {
          elem.properties[iP]->readNextBigEndian(inStream);
        }
      }
    }
  }

  // === Writing ===


  /**
   * @brief write a PLY file to an output stream
   *
   * @param outStream
   */
  void writePLY(std::ostream& outStream) {

    writeHeader(outStream);

    // Write all elements
    for (Element& e : elements) {
      if (outputDataFormat == DataFormat::Binary) {
        if (!isLittleEndian()) {
          throw std::runtime_error("binary writing assumes little endian system");
        }
        e.writeDataBinary(outStream);
      } else if (outputDataFormat == DataFormat::BinaryBigEndian) {
        if (!isLittleEndian()) {
          throw std::runtime_error("binary writing assumes little endian system");
        }
        e.writeDataBinaryBigEndian(outStream);
      } else if (outputDataFormat == DataFormat::ASCII) {
        e.writeDataASCII(outStream);
      }
    }
  }


  /**
   * @brief Write out a header for a file
   *
   * @param outStream
   */
  void writeHeader(std::ostream& outStream) {

    // Magic line
    outStream << "ply\n";

    // Type line
    outStream << "format ";
    if (outputDataFormat == DataFormat::Binary) {
      outStream << "binary_little_endian ";
    } else if (outputDataFormat == DataFormat::BinaryBigEndian) {
      outStream << "binary_big_endian ";
    } else if (outputDataFormat == DataFormat::ASCII) {
      outStream << "ascii ";
    }

    // Version number
    outStream << majorVersion << "." << minorVersion << "\n";

    // Write comments
    bool hasHapplyComment = false;
    std::string happlyComment = "Written with hapPLY (https://github.com/nmwsharp/happly)";
    for (const std::string& comment : comments) {
      if (comment == happlyComment) hasHapplyComment = true;
      outStream << "comment " << comment << "\n";
    }
    if (!hasHapplyComment) {
      outStream << "comment " << happlyComment << "\n";
    }

    // Write obj_info comments
    for (const std::string& comment : objInfoComments) {
      outStream << "obj_info " << comment << "\n";
    }

    // Write elements (and their properties)
    for (Element& e : elements) {
      e.writeHeader(outStream);
    }

    // End header
    outStream << "end_header\n";
  }
};

} // namespace happly

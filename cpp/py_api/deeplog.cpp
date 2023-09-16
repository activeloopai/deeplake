#include <pybind11/pybind11.h>

//int add(int i, int j) {
//    return i + j;
//}
//
//PYBIND11_MODULE(deeplake, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring
//
//    m.def("add", &add, "A function that adds two numbers");
//}

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

PYBIND11_MODULE(_deeplake, m) {
    auto m_deeplog = m.def_submodule("_deeplog");

    pybind11::class_<Pet>(m_deeplog, "Pet", "A Pet")
            .def(pybind11::init<const std::string &>())
            .def("setName", &Pet::setName, "Set the pet's name")
            .def("getName", &Pet::getName, "Get the pet's name");
}
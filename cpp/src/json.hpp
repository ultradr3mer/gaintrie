#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cle {

// minimal JSON value; objects keep their key order (like Python dicts)
struct Json {
    enum Type { NUL, BOOL, INT, FLOAT, STRING, ARRAY, OBJECT };
    Type type = NUL;
    bool b = false;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool v) : type(BOOL), b(v) {}
    Json(int v) : type(INT), i(v) {}
    Json(long v) : type(INT), i(v) {}
    Json(long long v) : type(INT), i(v) {}
    Json(unsigned long v) : type(INT), i(int64_t(v)) {}
    Json(unsigned long long v) : type(INT), i(int64_t(v)) {}
    Json(double v) : type(FLOAT), f(v) {}
    Json(const char* v) : type(STRING), s(v) {}
    Json(std::string v) : type(STRING), s(std::move(v)) {}
    static Json array() { Json j; j.type = ARRAY; return j; }
    static Json object() { Json j; j.type = OBJECT; return j; }

    const Json* find(const std::string& key) const;
    const Json& at(const std::string& key) const;
    Json& set(const std::string& key, Json v);  // replaces or appends
    Json& push(Json v) { arr.push_back(std::move(v)); return arr.back(); }
    int64_t as_int() const;
    const std::string& as_str() const;
};

Json json_parse(const std::string& text);
// as Python json.dumps: indent < 0 -> one line with ", " / ": ", else indent spaces per level; ensure_ascii
std::string json_dump(const Json& v, int indent = -1);
// repr(float) of Python: shortest round-trip digits
std::string py_float(double x);
// round(x, digits) of Python
double py_round(double x, int digits);

}  // namespace cle

#include "json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace cle {

const Json* Json::find(const std::string& key) const {
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const Json& Json::at(const std::string& key) const {
    const Json* v = find(key);
    if (!v) throw std::runtime_error("json: missing key " + key);
    return *v;
}

Json& Json::set(const std::string& key, Json v) {
    for (auto& kv : obj)
        if (kv.first == key) return kv.second = std::move(v);
    obj.emplace_back(key, std::move(v));
    return obj.back().second;
}

int64_t Json::as_int() const {
    if (type != INT) throw std::runtime_error("json: integer expected");
    return i;
}

const std::string& Json::as_str() const {
    if (type != STRING) throw std::runtime_error("json: string expected");
    return s;
}

namespace {

struct Parser {
    const std::string& t;
    size_t p = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error("json: " + std::string(what) + " at offset " + std::to_string(p));
    }
    void ws() {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) p++;
    }
    bool lit(const char* s) {
        size_t n = std::char_traits<char>::length(s);
        if (t.compare(p, n, s) != 0) return false;
        p += n;
        return true;
    }
    unsigned hex4() {
        if (p + 4 > t.size()) fail("bad \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; k++) {
            char c = t[p++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad \\u escape");
        }
        return v;
    }
    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    std::string str() {
        if (t[p] != '"') fail("string expected");
        p++;
        std::string out;
        while (true) {
            if (p >= t.size()) fail("unterminated string");
            char c = t[p++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p >= t.size()) fail("unterminated string");
            switch (char e = t[p++]) {
                case '"': case '\\': case '/': out += e; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && t.compare(p, 2, "\\u") == 0) {
                        size_t save = p;
                        p += 2;
                        unsigned lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else p = save;
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json num() {
        size_t s = p;
        if (t[p] == '-') p++;
        bool real = false;
        while (p < t.size()) {
            char c = t[p];
            if (c >= '0' && c <= '9') p++;
            else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') real = true, p++;
            else break;
        }
        std::string n = t.substr(s, p - s);
        if (n.empty() || n == "-") fail("bad number");
        if (!real) {
            int64_t v;
            auto r = std::from_chars(n.data(), n.data() + n.size(), v);
            if (r.ec == std::errc() && r.ptr == n.data() + n.size()) return Json(static_cast<long long>(v));
        }
        char* end;
        double d = std::strtod(n.c_str(), &end);
        if (end != n.c_str() + n.size()) fail("bad number");
        return Json(d);
    }
    Json value() {
        ws();
        if (p >= t.size()) fail("unexpected end");
        char c = t[p];
        if (c == '{') {
            p++;
            Json o = Json::object();
            ws();
            if (p < t.size() && t[p] == '}') return p++, o;
            while (true) {
                ws();
                if (p >= t.size()) fail("unexpected end");
                std::string k = str();
                ws();
                if (p >= t.size() || t[p] != ':') fail("':' expected");
                p++;
                o.set(k, value());
                ws();
                if (p < t.size() && t[p] == ',') p++;
                else if (p < t.size() && t[p] == '}') return p++, o;
                else fail("',' or '}' expected");
            }
        }
        if (c == '[') {
            p++;
            Json a = Json::array();
            ws();
            if (p < t.size() && t[p] == ']') return p++, a;
            while (true) {
                a.push(value());
                ws();
                if (p < t.size() && t[p] == ',') p++;
                else if (p < t.size() && t[p] == ']') return p++, a;
                else fail("',' or ']' expected");
            }
        }
        if (c == '"') return Json(str());
        if (lit("true")) return Json(true);
        if (lit("false")) return Json(false);
        if (lit("null")) return Json(nullptr);
        if (lit("NaN")) return Json(std::nan(""));
        if (lit("Infinity")) return Json(HUGE_VAL);
        if (lit("-Infinity")) return Json(-HUGE_VAL);
        return num();
    }
};

void dump_str(std::string& out, const std::string& s) {
    static const char* HEX = "0123456789abcdef";
    auto u = [&](unsigned v) {
        out += "\\u";
        for (int k = 12; k >= 0; k -= 4) out += HEX[(v >> k) & 15];
    };
    out += '"';
    for (size_t k = 0; k < s.size();) {
        unsigned char c = s[k];
        if (c < 0x80) {
            k++;
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else if (c == '\t') out += "\\t";
            else if (c == '\b') out += "\\b";
            else if (c == '\f') out += "\\f";
            else if (c < 0x20 || c == 0x7f) u(c);
            else out += char(c);
            continue;
        }
        int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
        unsigned cp = c & (0x3F >> n);
        for (int j = 1; j <= n && k + j < s.size(); j++) cp = (cp << 6) | (static_cast<unsigned char>(s[k + j]) & 0x3F);
        k += n + 1;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            u(0xD800 + (cp >> 10));
            u(0xDC00 + (cp & 0x3FF));
        } else {
            u(cp);
        }
    }
    out += '"';
}

void dump(std::string& out, const Json& v, int indent, int level) {
    auto nl = [&](int l) {
        out += '\n';
        out.append(size_t(indent * l), ' ');
    };
    switch (v.type) {
        case Json::NUL: out += "null"; break;
        case Json::BOOL: out += v.b ? "true" : "false"; break;
        case Json::INT: out += std::to_string(v.i); break;
        case Json::FLOAT:
            out += std::isnan(v.f) ? "NaN" : std::isinf(v.f) ? (v.f > 0 ? "Infinity" : "-Infinity") : py_float(v.f);
            break;
        case Json::STRING: dump_str(out, v.s); break;
        case Json::ARRAY:
        case Json::OBJECT: {
            bool obj = v.type == Json::OBJECT;
            size_t n = obj ? v.obj.size() : v.arr.size();
            out += obj ? '{' : '[';
            for (size_t k = 0; k < n; k++) {
                if (k) out += indent < 0 ? ", " : ",";
                if (indent >= 0) nl(level + 1);
                if (obj) {
                    dump_str(out, v.obj[k].first);
                    out += ": ";
                    dump(out, v.obj[k].second, indent, level + 1);
                } else {
                    dump(out, v.arr[k], indent, level + 1);
                }
            }
            if (n && indent >= 0) nl(level);
            out += obj ? '}' : ']';
            break;
        }
    }
}

}  // namespace

Json json_parse(const std::string& text) {
    Parser ps{text};
    Json v = ps.value();
    ps.ws();
    if (ps.p != text.size()) ps.fail("trailing data");
    return v;
}

std::string json_dump(const Json& v, int indent) {
    std::string out;
    dump(out, v, indent, 0);
    return out;
}

std::string py_float(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    std::string sign = s[0] == '-' ? "-" : "";
    if (!sign.empty()) s.erase(0, 1);
    size_t e = s.find('e');
    int exp = std::atoi(s.c_str() + e + 1);
    std::string digits = s.substr(0, e);
    if (digits.size() > 1) digits.erase(1, 1);
    int nd = int(digits.size());
    std::string out;
    if (exp >= -4 && exp < 16) {
        if (exp >= 0) {
            if (nd <= exp + 1) out = digits + std::string(size_t(exp + 1 - nd), '0') + ".0";
            else out = digits.substr(0, size_t(exp + 1)) + "." + digits.substr(size_t(exp + 1));
        } else {
            out = "0." + std::string(size_t(-exp - 1), '0') + digits;
        }
    } else {
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out = digits.substr(0, 1) + (nd > 1 ? "." + digits.substr(1) : "") + eb;
    }
    return sign + out;
}

double py_round(double x, int digits) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%.*f", digits, x);
    return std::strtod(buf, nullptr);
}

}  // namespace cle

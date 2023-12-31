/**
 ==============================================================================
 Copyright 2019 - 2024, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
 Rationale:

 This file provides utility functions for filesystem manipulations on
 Unix-like operating systems. I wrote this code in 2019 and was used in
 Mara3p/ At this time it is not a part of the VAPOR library, but it could
 become a part of the library if the need arises.
 ==============================================================================
*/




#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <cxxabi.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>




// ============================================================================
namespace vapor::filesystem
{
    inline std::vector<std::string> listdir(std::string path);
    inline std::vector<std::string> split(std::string path);
    inline std::string join(std::vector<std::string> parts);
    inline std::string extension(std::string path);
    inline std::string parent(std::string path);
    inline void require_dir(std::string path);
    inline int remove_file(std::string path);
    inline int remove_recurse(std::string path);
    inline bool isfile(std::string path);
    inline bool isdir(std::string path);

    template<typename... Args>
    std::string join(Args... args)
    {
        return join(std::vector<std::string>{args...});
    }
}




// ============================================================================
std::vector<std::string> vapor::filesystem::listdir(std::string path)
{
    std::vector<std::string> res;

    if (auto dir = opendir(path.data()))
    {
        while (auto f = readdir(dir))
        {
            if (f->d_name[0] != '.')
            {
                res.push_back(f->d_name);
            }
        }
        closedir(dir);
    }
    else
    {
        throw std::invalid_argument("no such directory " + path);
    }
    return res;
}

std::vector<std::string> vapor::filesystem::split(std::string path)
{
    auto remaining = path;
    auto dirs = std::vector<std::string>();

    while (true)
    {
        auto slash = remaining.find('/');

        if (slash == std::string::npos)
        {
            dirs.push_back(remaining);
            break;
        }
        dirs.push_back(remaining.substr(0, slash));
        remaining = remaining.substr(slash + 1);
    }
    return dirs;
}

std::string vapor::filesystem::join(std::vector<std::string> parts)
{
    auto res = std::string();

    for (auto part : parts)
    {
        res += "/" + part;
    }
    return res.empty() ? res : res.substr(1);
}

std::string vapor::filesystem::extension(std::string path)
{
    auto dot = path.rfind('.');

    if (dot != std::string::npos)
    {
        return path.substr(dot);
    }
    return "";
}

std::string vapor::filesystem::parent(std::string path)
{
    std::string::size_type lastSlash = path.find_last_of("/");
    return path.substr(0, lastSlash);
}

void vapor::filesystem::require_dir(std::string path)
{
    auto partial = std::string(".");

    for (auto dir : split(path))
    {
        partial += "/" + dir;
        mkdir(partial.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

int vapor::filesystem::remove_file(std::string path)
{
    return std::remove(path.c_str());
}

int vapor::filesystem::remove_recurse(std::string path)
{
    /**
     * Adapted from:
     * 
     * https://shorturl.at/blDPY
     * 
     * Uses methods:
     * opendir, closedir, readdir, rmdir, unlink, stat, S_ISDIR
     * 
     * Uses structs:
     * dirent, statbuf
     * 
     */

    int res = -1;

    if (auto d = opendir(path.data()))
    {
        struct dirent *p;
        res = 0;

        while (! res && (p = readdir(d)))
        {
            if (! std::strcmp(p->d_name, ".") || ! std::strcmp(p->d_name, ".."))
            {
                continue;
            }

            int res2 = -1;
            auto buf = std::string(path.size() + std::strlen(p->d_name) + 2, 0);

            std::snprintf(&buf[0], buf.size(), "%s/%s", path.data(), p->d_name);
            struct stat statbuf;

            if (! stat(buf.data(), &statbuf))
            {
                if (S_ISDIR(statbuf.st_mode))
                {
                    res2 = remove_recurse(buf.data());
                }
                else
                {
                    res2 = unlink(buf.data());
                }
            }
            res = res2;
        }
        closedir(d);
    }

    if (! res)
    {
        res = rmdir(path.data());
    }
    return res;
}

bool vapor::filesystem::isfile(std::string path)
{
    struct stat s;

    if (stat(path.data(), &s) == 0)
    {
        return s.st_mode & S_IFREG;
    }
    return false;
}

bool vapor::filesystem::isdir(std::string path)
{
    struct stat s;

    if (stat(path.data(), &s) == 0)
    {
        return s.st_mode & S_IFDIR;
    }
    return false;
}

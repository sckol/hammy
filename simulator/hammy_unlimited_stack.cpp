#include <iostream>
#include <cstdlib>

int main() {
    std::string script = R"(
#!/bin/bash
ulimit -s unlimited
./hammy
)";
    int result = system(script.c_str());
}

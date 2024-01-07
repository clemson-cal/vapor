/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

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
================================================================================
*/

#define VAPOR_USE_SHARED_PTR_ALLOCATOR
#include "vapor/sim.hpp"
#include <unistd.h> // usleep, for demo simulation class




struct Config
{
    int num_zones = 100;
    double tfinal = 1.0;
    double cpi = 0.0;
    std::vector<int> ts;
};
VISITABLE_STRUCT(Config, num_zones, tfinal, cpi, ts);




struct State
{
    double time;
    int iteration;
    vapor::memory_backed_array_t<1, double, std::shared_ptr> u;
};
VISITABLE_STRUCT(State, time, iteration, u);




using DiagnosticData = vapor::memory_backed_array_t<1, double, std::shared_ptr>;




class DemoSimulation : public vapor::Simulation<Config, State, DiagnosticData>
{
public:
    double get_time(const State& state) const override
    {
        return state.time;
    }
    virtual vapor::uint get_iteration(const State& state) const override
    {
        return state.iteration;
    }
    void initial_state(State& state) const override
    {
        state.time = 0.0;
        state.iteration = 0;
        state.u = vapor::zeros<double>(vapor::uvec(config.num_zones)).cache();
    }
    void update(State& state) const override
    {
        state.time += 0.0085216;
        state.iteration += 1;
        usleep(10000);
    }
    bool should_continue(const State& state) const override
    {
        return state.time < config.tfinal;
    }
    double checkpoint_interval() const override
    { 
        return config.cpi;
    }
};




int main(int argc, const char **argv)
{
    try {
        auto sim = DemoSimulation();
        return run(argc, argv, sim);
    }
    catch (const std::exception& e) {
        printf("[error]: %s\n", e.what());
    }
    return 0;
}

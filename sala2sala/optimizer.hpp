#ifndef SALA2SALA_OPTIMIZER_HPP_INCLUDED
#   define SALA2SALA_OPTIMIZER_HPP_INCLUDED

#   include <sala/program.hpp>


struct Optimizer
{
    Optimizer(sala::Program& P);

    void run();

    sala::Program& program() { return program_; }

private:

    sala::Program& program_;
};


#endif

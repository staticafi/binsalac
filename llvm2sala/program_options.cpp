#include <llvm2sala/program_options.hpp>
#include <llvm2sala/program_info.hpp>
#include <utility/assumptions.hpp>

program_options::program_options(int argc, char* argv[])
    : program_options_default(argc, argv)
{
    add_option("input", "Pathname to the input LLVM file (.sim.ll).", "1");
    add_option("output", "Pathname to the output Sala file (.json).", "1");
    add_option("entry", "A name of the entry function of the Sala program. The default name is 'main'.", "1");
    add_option("jsonc", "When specified a Sala file with dbg lines (.jsonc) is saved as well.", "0");
}

static program_options_ptr  global_program_options;

void initialise_program_options(int argc, char* argv[])
{
    ASSUMPTION(!global_program_options.operator bool());
    global_program_options = program_options_ptr(new program_options(argc,argv));
}

program_options_ptr get_program_options()
{
    ASSUMPTION(global_program_options.operator bool());
    return global_program_options;
}

std::ostream& operator<<(std::ostream& ostr, program_options_ptr const& options)
{
    ASSUMPTION(options.operator bool());
    options->operator<<(ostr);
    return ostr;
}

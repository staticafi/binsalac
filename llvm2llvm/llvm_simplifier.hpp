#ifndef LLVM2LLVM_LLVM_SIMPLIFIER_HPP_INCLUDED
#   define LLVM2LLVM_LLVM_SIMPLIFIER_HPP_INCLUDED

#   include <filesystem>


void simplify_llvm_file(std::filesystem::path const& src_llvm_file_pathname, std::filesystem::path const& dst_llvm_file_pathname);


#endif

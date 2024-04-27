#!/usr/bin/env python3
import subprocess
import sys
import os
import time


class Salac:
    def __init__(self) -> None:
        self.script_dir = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))

        self.dist_dir = None
        self.tool_ext = None
        self.detect_salac_binaries_in_dir(self.script_dir)

        self.input_path = None
        self.output_dir = os.path.abspath(os.getcwd())
        self.rename = None
        self.entry_function = 'main'
        self.opt_level = 0
        self.use_m32 = False
        self.options = []
        self.verbose = False

    def detect_salac_binaries_in_dir(self, dir : str) -> bool:
        if self.dist_dir is None:
            for path in [
                    os.path.join(dir, "sala2sala.exe"),
                    os.path.join(dir, "sala2sala"),
                    ]:
                if os.path.isfile(path):
                    self.dist_dir = os.path.normpath(os.path.dirname(path))
                    self.tool_ext = os.path.splitext(dir)[1]
                    return True
        return False

    def log(self, message : str, brief_message: str = None, end='\n') -> None:
        if self.verbose:
            print(">>> " + message, end=end, flush=True)
        elif brief_message is not None:
            print(brief_message, end="", flush=True)

    def _execute(self, command_and_args, timeout_ = None):
        cmd = [x for x in command_and_args if len(x) > 0]
        self.log(" ".join(cmd), end=' ')
        return subprocess.run(cmd, timeout=timeout_).returncode == 0

    def compile(self) -> str:
        in_dir = os.path.dirname(self.input_path)
        in_name, in_ext = os.path.splitext(os.path.basename(self.input_path))
        in_ext = (".sim" if os.path.splitext(in_name)[1] == ".sim" else "") + in_ext
        out_name = in_name if self.rename is None else self.rename

        if in_ext.lower() in [".c", ".i"]:
            self.log("C -> llvm", end=' ')
            t0 = time.time()
            if self._execute(
                    [ "clang", "-O" + str(self.opt_level), "-g", "-S", "-emit-llvm", "-Wno-everything", "-fbracket-depth=1024",
                        ("-m32" if self.use_m32 is True else ""),
                        os.path.join(in_dir, in_name + in_ext),
                        "-o", os.path.join(self.output_dir, in_name + ".ll"),
                        ],
                    None) is False:
                raise Exception("Translation from C to LLVM has failed: " + os.path.join(in_dir, in_name + in_ext))
            t1 = time.time()
            self.log("Done[%ds]" % int(round(t1 - t0)))
            in_dir = self.output_dir
            in_ext = ".ll"

        if in_ext.lower() == ".ll":
            self.log("llvm -> sim.llvm", end=' ')
            t0 = time.time()
            if self._execute(
                    [ os.path.join(self.dist_dir, "llvm2llvm" + self.tool_ext),
                        "--input", os.path.join(in_dir, in_name + in_ext),
                        "--output", os.path.join(self.output_dir, in_name + ".sim.ll")
                        ],
                    None) is False:
                raise Exception("Simplification of LLVM has failed: " + os.path.join(in_dir, in_name + in_ext))
            t1 = time.time()
            self.log("Done[%ds]" % int(round(t1 - t0)))
            in_dir = self.output_dir
            in_ext = ".sim.ll"

        if in_ext.lower() == ".sim.ll":
            self.log("sim.llvm -> raw.json", end=' ')
            t0 = time.time()
            if self._execute(
                    [ os.path.join(self.dist_dir, "llvm2sala" + self.tool_ext),
                        "--input", os.path.join(in_dir, in_name + in_ext),
                        "--output", os.path.join(self.output_dir, in_name + ".raw.json"),
                        "--entry", self.entry_function
                        ] + self.options,
                    None) is False:
                raise Exception("Translation from LLVM to Sala has failed: " + os.path.join(in_dir, in_name + in_ext))
            t1 = time.time()
            self.log("Done[%ds]" % int(round(t1 - t0)))
            in_dir = self.output_dir
            in_ext = ".raw.json"

        if in_ext.lower() == ".raw.json":
            self.log("raw.json -> json", end=' ')
            t0 = time.time()
            if self._execute(
                    [ os.path.join(self.dist_dir, "sala2sala" + self.tool_ext),
                        "--input", os.path.join(in_dir, in_name + in_ext),
                        "--output", os.path.join(self.output_dir, out_name + ".json")
                        ] + self.options,
                    None) is False:
                raise Exception("Optimization of Sala code has failed: " + os.path.join(in_dir, in_name + in_ext))
            t1 = time.time()
            self.log("Done[%ds]" % int(round(t1 - t0)))
            in_dir = self.output_dir
            in_ext = ".json"

        return os.path.join(in_dir, out_name + in_ext)

    def _run(self) -> bool:
        start_time = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            os.chdir(self.output_dir)
            self.compile()
        except Exception as e:
            print("Stopped[%ds]" % int(round(time.time() - start_time)), flush=True)
            return False
        self.log("Done[%ds]" % int(round(time.time() - start_time)))
        return True

    def run(self) -> bool:
        i = 1
        while (i < len(sys.argv)):
            arg = sys.argv[i]

            if arg == "--help":
                self.help()
                return True
            if arg == "--version":
                self.version()
                return True

            if arg == "--verbose":
                self.verbose = True
            elif arg == "--input" and i+1 < len(sys.argv) and os.path.isfile(sys.argv[i+1]):
                self.input_path = os.path.normpath(os.path.abspath(sys.argv[i+1]))
                i += 1
            elif arg == "--output" and i+1 < len(sys.argv) and not os.path.isfile(sys.argv[i+1]):
                self.output_dir = os.path.normpath(os.path.abspath(sys.argv[i+1]))
                i += 1
            elif arg == "--rename" and i+1 < len(sys.argv) and not os.path.isfile(sys.argv[i+1]):
                self.rename = sys.argv[i+1]
                i += 1
            elif arg == "--entry" and i+1 < len(sys.argv) and not os.path.isfile(sys.argv[i+1]):
                self.entry_function = sys.argv[i+1]
                i += 1
            elif arg == "--bin" and i+1 < len(sys.argv) and not os.path.isfile(sys.argv[i+1]):
                self.detect_salac_binaries_in_dir(os.path.normpath(os.path.abspath(sys.argv[i+1])))
                i += 1
            elif arg == "--opt" and i+1 < len(sys.argv) and sys.argv[i+1].isnumeric():
                self.opt_level = min(2, max(0, int(sys.argv[i+1])))
                i += 1
            elif arg == "--m32":
                self.use_m32 = True
            else:
                self.options.append(arg)

            i += 1

        if self.input_path is None:
            raise Exception("Cannot find the input file.")
        if self.dist_dir is None:
            raise Exception("Cannot find the Salac's binaries.")

        return self._run()
    
    def help(self) -> None:
        print("Salac usage")
        print("===========")
        print("help                 Prints this help message.")
        print("version              Prints the version string.")
        print("input <PATH>         A pathname of a C/LLVM program to be compiled to Sala.")
        print("output <PATH>        An output directory. If not specified, then the current")
        print("                     directory is used.")
        print("rename <name>        A name, without extension, for the resulting Sala program.")
        print("                     The '.json' extension will be added automatically. If not")
        print("                     specified, then the name of the input program is used,")
        print("                     with the extension changed to '.json'.")
        print("entry <name>         Allows to specify a custom entry function of the program.")
        print("                     The default name is 'main'.")
        print("opt <0|1|2>          An optimization level for translation from C to Sala.")
        print("                     If not specified, then the level is 0.")
        print("m32                  When specified, the source C file will be compiled for")
        print("                     32-bit machine (cpu). Otherwise, 64-bit machine is assumed.")
        print("verbose              When specified, the script prints information about the")
        print("                     computation.")
        print("\nNext follows a listing of options of tools called from this script. When they are")
        print("passed to this script they will automatically be propagated to the corresponding tool.")

        print("\nThe options of the 'llvm2llvm' tool:")
        self._execute([ os.path.join(self.dist_dir, "llvm2llvm" + self.tool_ext), "--help"], None)
        print("\nThe options of the 'llvm2sala' tool:")
        self._execute([ os.path.join(self.dist_dir, "llvm2sala" + self.tool_ext), "--help"], None)
        print("\nThe options of the 'sala2sala' tool:")
        self._execute([ os.path.join(self.dist_dir, "sala2sala" + self.tool_ext), "--help"], None)

    def version(self) -> None:
        self._execute([ os.path.join(self.dist_dir, "sala2sala" + self.tool_ext), "--version"], None)


if __name__ == "__main__":
    old_cwd = os.getcwd()
    exit_code = 0
    try:
        exit_code = 0 if Salac().run() is True else 1
    except Exception as e:
        exit_code = 1
        print("ERROR: " + str(e), flush=True)
    finally:
        os.chdir(old_cwd)
    exit(exit_code)

import subprocess

def get_stat_sle(path:str, exe:str):
    with open(path, "w") as file:
        for i in range(1, 70 * 100, 100):
            subprocess.run(["g++", exe, "-D NUM_THREADS=40", f"-D CHUNCK_SIZE={i}", "-D TYPE=guided", "-o", "out.exe", "-fopenmp"])
            out = subprocess.run(["./out.exe"], stdout=subprocess.PIPE)
            file.write(str(float(out.stdout)) + ' ')
            print(out.stdout)


get_stat_sle("data/static1.txt", "SLE3.cpp")


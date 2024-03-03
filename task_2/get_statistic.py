import subprocess

with open("stat.txt", "w") as file:
    for i in range(1, 10**6, 100):
        subprocess.run(["g++", "SLE3.cpp", "-D NUM_THREADS=40", f"-D CHUNCK_SIZE={i}", "-D TYPE=static", "-o", "out.exe"])
        out = subprocess.run(["./out.exe"], stdout=subprocess.PIPE)
        file.write(str(float(out.stdout)) + ' ')

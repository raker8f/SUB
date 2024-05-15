import subprocess
#subprocess.run(['java', '-jar', '/workspaces/SUB/SUB.jar'])
try:
    output = subprocess.check_output(['java', '-jar', '/workspaces/SUB/SUB.jar'], text=True)
    out = output.strip().splitlines()[-1]
    out = eval(out.strip())
    out = out[0]
    print(out)
except subprocess.CalledProcessError as e:
    print("Command execution failed:", e)
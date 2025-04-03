import paramiko
import os

hostname = 'ubuntupi.local'
username = 'peder'
password = 'kristian'
samples = 31250

def remote_sample_adc(file_name):
    command = f'sudo ./lab1/adc_sampler {samples} ./lab1/{file_name}.bin'

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    try:
        stdin, stdout, stderr = client.exec_command(command)
        print(stdout.read().decode('utf-8'))
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        client.close()

def download_file(remote_dir, local_dir, remote_file_name, local_file_name):
    remote_file = f"{remote_dir}/{remote_file_name}.bin"
    local_file = f"{local_dir}/{local_file_name}.bin"
    os.makedirs(local_dir, exist_ok=True)
    try:
        transport = paramiko.Transport((hostname, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(remote_file, local_file)
        print(f"File downloaded successfully to {local_file}")
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error: {e}")

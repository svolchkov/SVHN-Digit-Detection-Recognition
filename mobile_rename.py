import os
counter = 5
for f in os.listdir("mobile"):
    if f.startswith("IMG"):
        new_file_name = os.path.join("mobile",str(counter) + ".jpg")
        old_file_name = os.path.join("mobile",f)
        counter += 1
        os.rename(old_file_name,new_file_name)
    

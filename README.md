# **AES-128 Encryption & Decryption with CBC Mode in Python** 🔐  

## **📌 Project Overview**  
This project implements the **Advanced Encryption Standard (AES)** in Python, covering both **encryption and decryption** mechanisms. AES is a widely used symmetric encryption algorithm that ensures secure data transmission. The project includes **AES key expansion, S-box transformations, matrix operations, and Cipher Block Chaining (CBC) mode encryption.**  

Additionally, this repository includes **two PDFs (`C-HW3.pdf` and `C-HW4.pdf`)**, which contain homework assignments related to encryption and decryption. The implemented **Python script ensures that encrypted or decrypted results match the expected outputs from these assignments for verification.**  

---

## **🚀 Features**  
✅ **AES Encryption & Decryption** – Implements **AES-128** with a **10-round key expansion**.  
✅ **Key Expansion** – Dynamically generates round keys using substitution and shift operations.  
✅ **Substitution Layer (S-box & Inverse S-box)** – Uses pre-defined lookup tables for byte substitution.  
✅ **Matrix Transformations** – Includes **Shift Rows, Mix Columns, and Add Round Key** operations.  
✅ **Galois Field Multiplication** – Ensures **data diffusion** using polynomial multiplication.  
✅ **AES in CBC Mode** – Uses **XOR chaining** for added security and **protection against pattern recognition**.  
✅ **Padding & Unpadding** – Handles variable-length input for seamless encryption and decryption.  
✅ **Homework Verification** – The script is designed to **process `C-HW3.pdf` and `C-HW4.pdf` correctly**, ensuring the AES implementation produces the expected encrypted and decrypted outputs.  
✅ **Configurable Encryption/Decryption Parameters** – The encryption and decryption values **can be adjusted** at the **end of the script** (see **last 20-30 lines of `aes_encryption_decryption.py`**).  
✅ **Pythonic Implementation** – **Clean, modular, and well-documented** Python code.  

---

## **💡 Project Impact**  
This project enhances **cryptography skills**, demonstrating expertise in **data security, encryption algorithms, and secure key management**. It can be applied to:  
- **Secure communication protocols**  
- **Encrypted databases**  
- **Authentication mechanisms**  

---

## **🛠️ Tech Stack**  
🔹 **Programming Language**: Python  
🔹 **Libraries Used**: `secrets`, `binascii` (for secure key generation and encoding)  
🔹 **Encryption Standard**: AES-128 with CBC mode  
🔹 **Key Length**: 128-bit  

---

## **📖 How to Run the Code**  

### **🔹 Prerequisites**  
Ensure you have **Python 3.7+** installed. You can download it from [here](https://www.python.org/downloads/).  

### **🔹 Installation**  
Clone the repository to your local machine:  
```sh
git clone https://github.com/Mobeen-01/AES-128-Encryption-Decryption.git
cd AES-128-Encryption-Decryption
```
  
### **🔹 Running the Script**  
To encrypt and decrypt a file, run the following command:  
```sh
python aes_encryption_decryption.py
```

### **🔹 Adjusting Encryption & Decryption Values**  
At the **end of the script (`aes_encryption_decryption.py`)**, there is a section labeled **"AES Encryption & Decryption Testing"** where you can modify the input **data and key** as per your needs.  
- **Look for the last 20-30 lines of the script**  
- You can update values for `data` and `key` in both **AES standard mode and CBC mode**  
- Example section in the script:  
  ```python
  data = b'2C\xf6\xa8\x88Z0\x8d11\x98\xa2\xe07\x074'
  key  = b'\x2B\x7E\x15\x16\x28\xAE\xD2\xA6\xAB\xF7\x15\x88\x09\xCF\x4F\x3C'

  ```
  Modify these values to test different encryptions and decryptions.  

### **🔹 Expected Output**  
- The script will **encrypt** `C-HW3.pdf` and store the output as an encrypted file.  
- It will then **decrypt** `C-HW4.pdf` and verify the correctness of decryption.  

If the implementation is correct, the decrypted output should **match** the original content from the homework files.  

---

## **📂 Project Structure**  
```
📂 AES-128-Encryption-Decryption
 ├── 📜 C-HW3.pdf  # Homework file related to encryption
 ├── 📜 C-HW4.pdf  # Homework file related to decryption
 ├── 📝 aes_encryption_decryption.py  # Python script implementing AES-128
 ├── 📝 README.md  # Documentation
```

---

## **💡 Future Enhancements**  
🔹 Implement **AES-192 and AES-256** for more security.  
🔹 Add a **Graphical User Interface (GUI)** for ease of use.  
🔹 Extend support for **other encryption modes** like GCM.  

---

## **📬 Connect with Me**  
Feel free to reach out for **collaboration, feedback, or discussion** on cryptography-related projects! 🚀  

🔗 **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/yourusername)  
🔗 **GitHub:** [Your GitHub Profile](https://github.com/Mobeen-01)  

---

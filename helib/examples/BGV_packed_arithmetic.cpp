/* Copyright (C) 2019-2021 IBM Corp.
 * This program is Licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

// This is a sample program for education purposes only.
// It attempts to show the various basic mathematical
// operations that can be performed on both ciphertexts
// and plaintexts.

#include <iostream>

#include <helib/helib.h>

int main(int argc, char *argv[]) {
  /*  Example of BGV scheme  */

  // Plaintext prime modulus
  unsigned long p = 2;
  // Cyclotomic polynomial - defines phi(m)
  unsigned long m = 21845;
  // Number of columns of Key-Switching matrix (default = 2 or 3)
  unsigned long c = 2;

  std::vector<long> gens = {8996, 17477, 21591};
  std::vector<long> mvec = {17, 5, 257};

  std::cout << "\n*********************************************************";
  std::cout << "\n*         Basic Mathematical Operations Example         *";
  std::cout << "\n*         =====================================         *";
  std::cout << "\n*                                                       *";
  std::cout << "\n* This is a sample program for education purposes only. *";
  std::cout << "\n* It attempts to show the various basic mathematical    *";
  std::cout << "\n* operations that can be performed on both ciphertexts  *";
  std::cout << "\n* and plaintexts.                                       *";
  std::cout << "\n*                                                       *";
  std::cout << "\n*********************************************************";
  std::cout << std::endl;

  std::cout << "Initialising context object..." << std::endl;
  // Initialize context
  // This object will hold information about the algebra created from the
  // previously set parameters
  helib::Context context = helib::ContextBuilder<helib::BGV>()
                               .m(m)
                               .p(p)
                               .c(c)
                               .gens(gens)
                               .mvec(mvec)
                               .bootstrappable(true)
                               .buildCache(true)
                               .thinboot()
                               .build();

  // Print the context
  context.printout();
  std::cout << std::endl;

  // Print the security level
  std::cout << "Security: " << context.securityLevel() << std::endl;

  // Secret key management
  std::cout << "Creating secret key..." << std::endl;
  // Create a secret key associated with the context
  helib::SecKey secret_key(context);
  // Generate the secret key
  secret_key.GenSecKey();
  std::cout << "Generating key-switching matrices..." << std::endl;
  // Compute key-switching matrices that we need
  helib::addSome1DMatrices(secret_key);
  // Generate recrypt data for bootstrapping
  std::cout << "Generating recrypting data..." << std::endl;
  secret_key.genRecryptData();

  // Public key management
  // Set the secret key (upcast: SecKey is a subclass of PubKey)
  const helib::PubKey &public_key = secret_key;

  // Get the EncryptedArray of the context
  const helib::EncryptedArray &ea = context.getEA();

  // Get the number of slot (phi(m))
  long nslots = ea.size();
  std::cout << "Number of slots: " << nslots << std::endl;

  // Create a vector of long with nslots elements
  helib::Ptxt<helib::BGV> ptxt(context);
  // Set it with numbers 0..nslots - 1
  // ptxt = [0] [1] [2] ... [nslots-2] [nslots-1]
  for (int i = 0; i < ptxt.size(); ++i) {
    ptxt[i] = i;
  }

  // Print the plaintext
  std::cout << "Initial Plaintext: " << ptxt << std::endl;

  // Create a ciphertext object
  helib::Ctxt ctxt(public_key);
  // Encrypt the plaintext using the public_key
  public_key.Encrypt(ctxt, ptxt);

  /********** Operations **********/
  // Ciphertext and plaintext operations are performed
  // "entry-wise".

  // Square the ciphertext
  ctxt.multiplyBy(ctxt);
  ptxt.multiplyBy(ptxt);

  // Create a plaintext for decryption
  helib::Ptxt<helib::BGV> plaintext_result(context);
  // Decrypt the modified ciphertext
  secret_key.Decrypt(plaintext_result, ctxt);

  std::cout << "Operation: a*a = 0" << std::endl;
  // Print the decrypted plaintext
  std::cout << "Decrypted Result: " << plaintext_result << std::endl;
  // Print the plaintext version result, should be the same as the ctxt version
  std::cout << "Plaintext Result: " << ptxt << std::endl;

  // We can also add constants
  ctxt.addConstant(NTL::ZZX(1l));
  ptxt.addConstant(NTL::ZZX(1l));

  // And multiply by constants
  ctxt *= 1l;
  ptxt *= 1l;

  // We can also perform ciphertext-plaintext operations
  // Note: the output of this is also a ciphertext
  ctxt += ptxt;

  // Perform thin bootstrapping
  helib::Ctxt ctxt2(ctxt);
  public_key.thinReCrypt(ctxt2);

  // Decrypt the modified ciphertext into a new plaintext
  helib::Ptxt<helib::BGV> new_plaintext_result(context);
  secret_key.Decrypt(new_plaintext_result, ctxt2);

  std::cout << "Operation: Enc{(0 + 1)*1} + (0 + 1)*1" << std::endl;
  // Print the decrypted plaintext
  std::cout << "Decrypted Result: " << new_plaintext_result << std::endl;

  return 0;
}

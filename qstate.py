# quantum_grant_search_quantum.py

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt  # type: ignore
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Grant Data Handling
class Grant:
    def __init__(self, title, description, amount, location):
        self.title = title
        self.description = description
        self.amount = amount
        self.location = location

# Sample Grant Data
grants = [
    Grant("Climate Action Grant", "Funding for projects reducing carbon emissions.", 50000, "Boston"),
    Grant("Environmental Research Grant", "Support for environmental impact studies.", 75000, "New York"),
    Grant("Sustainability Initiative Grant", "Grants for sustainable development projects.", 100000, "Boston"),
    Grant("Green Technology Grant", "Funding for innovative green technologies.", 25000, "San Francisco"),
    Grant("Community Clean Energy Grant", "Support for community-based clean energy.", 60000, "Boston"),
    # Add more grants as needed
]

# Quantum Grant Searcher
class QuantumGrantSearcher:
    def __init__(self, grants):
        self.grants = grants
        self.num_grants = len(grants)
        self.num_qubits = int(np.ceil(np.log2(self.num_grants)))
        self.backend = AerSimulator()

    def encode_query(self, query):
        # Simulate quantum-compatible NLP to find matching grants
        query_terms = query.lower().split()
        matching_indices = []
        partial_match_scores = []

        for i, grant in enumerate(self.grants):
            grant_text = f"{grant.title} {grant.description} {grant.location}".lower()
            grant_terms = set(grant_text.split())
            common_terms = set(query_terms) & grant_terms
            match_score = len(common_terms) / len(set(query_terms))

            if match_score == 1.0:
                matching_indices.append(i)
            elif match_score > 0:
                partial_match_scores.append((i, match_score))

        return matching_indices, partial_match_scores

    def create_oracle(self, indices):
        oracle = QuantumCircuit(self.num_qubits)
        if not indices:
            return oracle.to_gate(label='Oracle')

        for index in indices:
            index_bin = format(index, f'0{self.num_qubits}b')
            index_bin = index_bin[::-1]  # Reverse for little-endian
            for qubit, bit in enumerate(index_bin):
                if bit == '0':
                    oracle.x(qubit)
            if self.num_qubits == 1:
                oracle.z(0)
            else:
                oracle.h(self.num_qubits - 1)
                oracle.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                oracle.h(self.num_qubits - 1)
            for qubit, bit in enumerate(index_bin):
                if bit == '0':
                    oracle.x(qubit)
        return oracle.to_gate(label='Oracle')

    def create_diffuser(self):
        diffuser = QuantumCircuit(self.num_qubits)
        diffuser.h(range(self.num_qubits))
        diffuser.x(range(self.num_qubits))
        if self.num_qubits == 1:
            diffuser.z(0)
        else:
            diffuser.h(self.num_qubits - 1)
            diffuser.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            diffuser.h(self.num_qubits - 1)
        diffuser.x(range(self.num_qubits))
        diffuser.h(range(self.num_qubits))
        return diffuser.to_gate(label='Diffuser')

    def search(self, query):
        matching_indices, partial_match_scores = self.encode_query(query)

        if matching_indices:
            indices_to_search = matching_indices
            print("Exact matches found. Performing quantum search...")
        elif partial_match_scores:
            # Sort grants by match score in descending order
            partial_match_scores.sort(key=lambda x: x[1], reverse=True)
            top_matches = [index for index, score in partial_match_scores[:3]]  # Get top 3 partial matches
            indices_to_search = top_matches
            print("No exact matches found. Performing quantum search on top potential candidates...")
        else:
            print("No matching grants found.")
            return

        # Initialize quantum circuit
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(qr, cr)

        # Apply Hadamard gates to create superposition
        qc.h(qr)

        # Prepare Oracle and Diffuser
        oracle = self.create_oracle(indices_to_search)
        diffuser = self.create_diffuser()

        # Determine the number of iterations
        num_iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** self.num_qubits / len(indices_to_search))))
        if num_iterations == 0:
            num_iterations = 1  # Ensure at least one iteration

        # Apply Grover's algorithm
        for _ in range(num_iterations):
            qc.append(oracle, qr)
            qc.append(diffuser, qr)

        # Measurement
        qc.measure(qr, cr)

        # Transpile the circuit for the backend
        transpiled_qc = transpile(qc, self.backend)

        # Execute the circuit
        job = self.backend.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Find the most frequent result
        max_count = max(counts.values())
        most_common_states = [state for state, count in counts.items() if count == max_count]

        # Decode the index (little-endian)
        result_state = most_common_states[0][::-1]
        grant_index = int(result_state, 2)
        best_grant = self.grants[grant_index]

        # Output the result
        print(f"\nThe most relevant grant based on your query '{query}' is:")
        print(f"Title: {best_grant.title}")
        print(f"Description: {best_grant.description}")
        print(f"Amount: ${best_grant.amount}")
        print(f"Location: {best_grant.location}")

        # If there were partial matches, list them
        if not matching_indices and partial_match_scores:
            print("\nOther potential candidate grants:")
            for index, score in partial_match_scores[:3]:
                grant = self.grants[index]
                print(f"- Title: {grant.title}, Match Score: {score:.2f}")

        # Plot histogram
        plot_histogram(counts)
        plt.show()

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Quantum Grant Search Tool using Grover\'s Algorithm')
    parser.add_argument('query', type=str, nargs='?', default='',
                        help='Search query to find relevant grants (e.g., "renewable energy Boston")')
    args = parser.parse_args()

    if not args.query:
        print("Please provide a search query.")
        sys.exit(1)

    # Initialize the Quantum Grant Searcher
    searcher = QuantumGrantSearcher(grants)

    # Perform the quantum search
    searcher.search(args.query)

if __name__ == "__main__":
    main()
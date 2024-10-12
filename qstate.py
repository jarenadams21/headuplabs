# quantum_grant_search_quantum.py

#! Imports and Dependencies
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt  # type: ignore
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

# System Equation:
# The quantum state vector evolves under the Grover operator G:
# |\psi_{k+1}\rangle = G |\psi_k\rangle
# where G = D * O
# - O is the Oracle operator
# - D is the Diffuser (Inversion about the mean) operator
# This equation governs the evolution of the quantum state in Grover's algorithm.

#! Grant Data Handling
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
    Grant("Eco-Friendly Transportation Grant", "Support for sustainable transportation initiatives.", 40000, "San Diego"),
    Grant("Recycling Innovation Grant", "Grants for innovative recycling technologies.", 35000, "Portland"),
    Grant("Sustainable Agriculture Grant", "Funding for sustainable farming practices.", 60000, "Austin"),
    Grant("Carbon Neutrality Grant", "Support for achieving carbon neutrality in organizations.", 70000, "Boston"),
    Grant("Little Onion Restaurant Grant", "Support for small businesses and restaurants in California and Nevada.", 5000, "Santa Ana"),
    Grant("Mike's Grant", "I am legit but also a scam, but I'll give you more! Give me business, now!", 10000, "Orange Grove"), #! TODO: Filter, controlling the 0 and 1 function..? (IBM video)
    Grant("Subnautic Travelers", "All sea-men and voyagers of the blue alike!", 100000, "Highwaters, LN"),
    Grant("A Time Ago", "Subsidizing Egyptian student housing and groceries", 3500, "Cairo, Egypt"),
    #! Fill dataset
]

# Quantum Grant Searcher
class QuantumGrantSearcher:
    def __init__(self, grants):
        self.grants = grants
        self.num_grants = len(grants)
        self.num_qubits = int(np.ceil(np.log2(self.num_grants)))
        self.backend = AerSimulator(method='statevector')  # Use statevector simulator for state tracking

    def encode_query(self, query):
        '''
        encode_query:
        [
                1. Query Processing: splits user query into lowercase terms
                2. Matching Logic: for each grant, compute the overlap between query terms and grant terms
                    i) Exact Match: Grants where all query terms are present
                    ii) Partial Match: Grants with some overlap, scored based on the proportion of matching terms
                Returns
                    <R1> matching_indices: indices of grants with exact matches
                    <R2> partial_match_scores: List of tuples containing grant indices and their respective match scores for partial matches
        ]
        '''
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
        '''
        create_oracle: Constructs the Oracle gate used in Grover's search, flipping the phase of states corresponding to target indices (matching grants)
        '''
        oracle = QuantumCircuit(self.num_qubits)
        if not indices:
            return oracle.to_gate(label='Oracle')

        for index in indices:
            index_bin = format(index, f'0{self.num_qubits}b')
            # Apply X gates to qubits where the index bit is '0'
            for qubit in range(self.num_qubits):
                if index_bin[qubit] == '0':
                    oracle.x(qubit)
            # Apply multi-controlled Z gate
            if self.num_qubits == 1:
                oracle.z(0)
            else:
                oracle.h(self.num_qubits - 1)
                oracle.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                oracle.h(self.num_qubits - 1)
            # Undo the X gates
            for qubit in range(self.num_qubits):
                if index_bin[qubit] == '0':
                    oracle.x(qubit)
        return oracle.to_gate(label='Oracle')

    def create_diffuser(self):
        '''
        create_diffuser: Implements Grover diffuser (inversion about mean), amplifying probability amplitudes of target states
        '''
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
        '''
        search: Performs the quantum search using Grover's algorithm
        '''
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

        # Initialize the statevector
        initial_state = Statevector.from_label('0' * self.num_qubits)

        # Apply Hadamard gates to create superposition
        state = initial_state.evolve(qc)

        # Mapping from state labels to grant titles
        state_to_grant = {}
        for i in range(len(self.grants)):
            state_label = format(i, f'0{self.num_qubits}b')
            state_to_grant[state_label] = self.grants[i].title

        # Collect probabilities for plotting
        probabilities_list = []
        probabilities = state.probabilities_dict()
        probabilities_list.append(probabilities)

        # Apply Grover's algorithm iterations
        for iteration in range(num_iterations):
            # Apply Oracle
            qc.append(oracle, qr)
            state = state.evolve(oracle)
            # Apply Diffuser
            qc.append(diffuser, qr)
            state = state.evolve(diffuser)
            # Compute probabilities according to Born's rule
            probabilities = state.probabilities_dict()
            probabilities_list.append(probabilities)

        # Measurement
        qc.measure(qr, cr)

        # Transpile the circuit for the backend
        transpiled_qc = transpile(qc, self.backend)

        # Execute the circuit
        job = self.backend.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Map counts to grant titles
        counts_grants = {}
        for state_str, count in counts.items():
            grant_index = int(state_str, 2)
            grant_title = self.grants[grant_index].title if grant_index < len(self.grants) else 'Unknown'
            counts_grants[grant_title] = counts_grants.get(grant_title, 0) + count

        # Find the most frequent result
        max_count = max(counts.values())
        most_common_states = [state for state, count in counts.items() if count == max_count]
        result_state = most_common_states[0]
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

        # Plot histogram of measurement results with grant titles
        plot_histogram(counts_grants)
        plt.show()

        # Plot probability currents
        # Prepare data for plotting
        iterations = range(len(probabilities_list))
        state_labels = ['{0:0{1}b}'.format(i, self.num_qubits) for i in range(2 ** self.num_qubits)]

        # For each state, collect probabilities over iterations
        state_probabilities = {state: [] for state in state_labels}
        for probs in probabilities_list:
            for state in state_labels:
                state_probabilities[state].append(probs.get(state, 0))

        # Plot probabilities for each state over iterations
        plt.figure(figsize=(12, 6))
        for state, probs in state_probabilities.items():
            grant_title = state_to_grant.get(state, "Unknown")
            plt.plot(iterations, probs, label=f'{grant_title}')
        plt.xlabel('Iteration')
        plt.ylabel('Probability')
        plt.title('Probability Currents Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# Main Function
def main():
    '''
    Functionality:
        i) Argument Parsing: Accepts a search query as a command-line argument
        ii) Validation: Ensures query is given, otherwise exits.
        iii) Execution: Creates a QuantumGrantSearcher and invokes the search on the query from arguments
    '''
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
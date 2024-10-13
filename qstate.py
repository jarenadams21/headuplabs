# quantum_grant_search_quantum.py

# Imports and Dependencies
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt  # type: ignore
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, Operator
from matplotlib import animation
from matplotlib.widgets import Button

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# System Equation:
# The quantum state vector evolves under the Grover operator G:
# |\psi_{k+1}\rangle = G |\psi_k\rangle
# where G = D * O
# - O is the Oracle operator
# - D is the Diffuser (Inversion about the mean) operator
# This equation governs the evolution of the quantum state in Grover's algorithm.
# The probability amplitudes evolve analogously to celestial bodies under gravitational influence,
# with amplitudes being 'pulled' towards the target states, similar to how masses influence each other in space.

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
    Grant("Eco-Friendly Transportation Grant", "Support for sustainable transportation initiatives.", 40000, "San Diego"),
    Grant("Recycling Innovation Grant", "Grants for innovative recycling technologies.", 35000, "Portland"),
    Grant("Sustainable Agriculture Grant", "Funding for sustainable farming practices.", 60000, "Austin"),
    Grant("Carbon Neutrality Grant", "Support for achieving carbon neutrality in organizations.", 70000, "Boston"),
    Grant("Little Onion Restaurant Grant", "Support for small businesses and restaurants in California and Nevada.", 5000, "Santa Ana"),
    Grant("Mike's Grant", "I am legit but also a scam, but I'll give you more! Give me business, now!", 10000, "Orange Grove"),
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
            1. Query Processing: Splits user query into lowercase terms and lemmatizes them.
            2. Matching Logic: For each grant, lemmatize the grant terms and compute the overlap with lemmatized query terms.
                i) Exact Match: Grants where all query terms are present after lemmatization.
                ii) Partial Match: Grants with some overlap, scored based on the proportion of matching terms.
            Returns:
                matching_indices: Indices of grants with exact matches.
                partial_match_scores: List of tuples containing grant indices and their respective match scores for partial matches.
        ]
        '''
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Remove punctuation from query terms and lemmatize them
        translator = str.maketrans('', '', string.punctuation)
        query_terms = nltk.word_tokenize(query.lower().translate(translator))
        lemmatized_query_terms = [lemmatizer.lemmatize(term) for term in query_terms]
        query_terms_set = set(lemmatized_query_terms)
        
        matching_indices = []
        partial_match_scores = []
        
        for i, grant in enumerate(self.grants):
            grant_text = f"{grant.title} {grant.description} {grant.location}".lower()
            # Remove punctuation from grant text and lemmatize the terms
            grant_text = grant_text.translate(translator)
            grant_terms = nltk.word_tokenize(grant_text)
            lemmatized_grant_terms = [lemmatizer.lemmatize(term) for term in grant_terms]
            
            grant_terms_set = set(lemmatized_grant_terms)
            common_terms = query_terms_set & grant_terms_set
            match_score = len(common_terms) / len(query_terms_set) if len(query_terms_set) > 0 else 0
            
            if match_score == 1.0:
                matching_indices.append(i)
            elif match_score > 0:
                partial_match_scores.append((i, match_score))
        
        return matching_indices, partial_match_scores

    def create_oracle(self, indices):
        '''
        create_oracle: Constructs the Oracle gate used in Grover's search, flipping the phase of states corresponding to target indices (matching grants).
        '''
        oracle = QuantumCircuit(self.num_qubits)
        if not indices:
            # No indices to flip; return identity oracle
            return oracle.to_gate()
        
        # Create an oracle that flips the phase of the states corresponding to indices
        oracle_matrix = np.identity(2 ** self.num_qubits)
        for index in indices:
            oracle_matrix[index][index] = -1
        
        # Convert the oracle matrix to an operator
        oracle_operator = Operator(oracle_matrix)
        oracle_gate = oracle_operator.to_instruction()
        return oracle_gate

    def create_diffuser(self):
        '''
        create_diffuser: Implements Grover diffuser (inversion about mean), amplifying probability amplitudes of target states.
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
        return diffuser.to_gate()

    def search(self, query):
        '''
        search: Performs the quantum search using Grover's algorithm.
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
        num_iterations = int(np.round(np.pi / 4 * np.sqrt(2 ** self.num_qubits / len(indices_to_search))))
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

            # Print the probability currents
            print(f"Iteration {iteration + 1}:")
            for state_str in sorted(probabilities):
                prob = probabilities[state_str]
                grant_title = state_to_grant.get(state_str, None)
                if grant_title:
                    print(f"  State {state_str} ({grant_title}): Probability = {prob:.4f}")
            print("-" * 50)

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
            if grant_index < len(self.grants):
                grant_title = self.grants[grant_index].title
                counts_grants[grant_title] = counts_grants.get(grant_title, 0) + count
            else:
                continue  # Skip invalid indices

        if counts_grants:
            # Find the most frequent result
            max_count = max(counts_grants.values())
            most_common_titles = [title for title, count in counts_grants.items() if count == max_count]
            best_grant_title = most_common_titles[0]
            # Find the grant with this title
            for grant in self.grants:
                if grant.title == best_grant_title:
                    best_grant = grant
                    break
        else:
            print("No valid grant found in measurement results.")
            return

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

        # Plot probability currents with stepper functionality
        self.plot_with_stepper(probabilities_list, state_to_grant, indices_to_search)

    def plot_with_stepper(self, probabilities_list, state_to_grant, indices_to_search):
        '''
        plot_with_stepper: Creates an interactive plot with stepper functionality to move back and forth between iterations.
        '''
        # Prepare data for plotting
        iterations = range(len(probabilities_list))
        grant_titles = [self.grants[i].title for i in range(len(self.grants))]
        grant_indices = list(range(len(self.grants)))
        num_frames = len(probabilities_list)
        current_frame = [0]  # Use a mutable object to allow modification inside nested functions

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(grant_indices, [0]*len(self.grants), tick_label=grant_titles)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Probability Evolution Over Iterations')
        plt.xticks(rotation=90)
        ax.set_xlabel(f'Iteration {current_frame[0]}')

        # Initialize the bars
        def init():
            y = []
            probabilities = probabilities_list[0]
            for grant_index in range(len(self.grants)):
                state_label = format(grant_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob in zip(bars, y):
                bar.set_height(prob)
                if prob > 0.01:
                    bar.set_color('green' if grant_index in indices_to_search else 'blue')
                else:
                    bar.set_color('grey')
            return bars

        init()

        # Function to update the bars for each frame
        def update(frame):
            y = []
            probabilities = probabilities_list[frame]
            for grant_index in range(len(self.grants)):
                state_label = format(grant_index, f'0{self.num_qubits}b')
                prob = probabilities.get(state_label, 0)
                y.append(prob)
            for bar, prob in zip(bars, y):
                bar.set_height(prob)
                if prob > 0.01:
                    bar.set_color('green' if grant_index in indices_to_search else 'blue')
                else:
                    bar.set_color('grey')
            ax.set_xlabel(f'Iteration {frame}')
            fig.canvas.draw_idle()

        # Button callback functions
        def next_iteration(event):
            if current_frame[0] < num_frames - 1:
                current_frame[0] += 1
                update(current_frame[0])

        def prev_iteration(event):
            if current_frame[0] > 0:
                current_frame[0] -= 1
                update(current_frame[0])

        # Play/Pause functionality
        is_playing = [False]

        def play_pause(event):
            is_playing[0] = not is_playing[0]
            if is_playing[0]:
                play_button.label.set_text('Pause')
                animate()
            else:
                play_button.label.set_text('Play')

        # Animation function
        def animate():
            if is_playing[0]:
                if current_frame[0] < num_frames - 1:
                    current_frame[0] += 1
                else:
                    current_frame[0] = 0  # Loop back to the start
                update(current_frame[0])
                fig.canvas.flush_events()
                fig.canvas.start_event_loop(0.5)
                animate()
            else:
                return

        # Add buttons
        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.05])
        ax_play = plt.axes([0.59, 0.02, 0.1, 0.05])

        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        play_button = Button(ax_play, 'Play')

        btn_prev.on_clicked(prev_iteration)
        btn_next.on_clicked(next_iteration)
        play_button.on_clicked(play_pause)

        plt.tight_layout()
        plt.show()

# Main Function
def main():
    '''
    Functionality:
        i) Argument Parsing: Accepts a search query as a command-line argument.
        ii) Validation: Ensures query is given, otherwise exits.
        iii) Execution: Creates a QuantumGrantSearcher and invokes the search on the query from arguments.
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

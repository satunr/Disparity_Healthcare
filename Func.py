import pandas as pd
import numpy as np
from collections import defaultdict

def decode_dataframe(df, encoders):
	"""
	Convert encoded categorical values back to their original semantic names.

	Parameters
	----------
	df : pandas.DataFrame
		Encoded dataframe.

	encoders : dict
		Dictionary returned by load_dataframe().

	Returns
	---------
	df_decoded : pandas.DataFrame
		Human-readable dataframe.
	"""

	df_decoded = df.copy()

	for col, mapping in encoders.items():
		# integer -> original string
		inverse_mapping = {v: k for k, v in mapping.items()}

		def decode(val):
			# Preserve missing values
			if val == -1:
				return "Unk"
			return inverse_mapping.get(val, val)
		df_decoded[col] = df_decoded[col].apply(decode)
    
	return df_decoded

def impute_dataframe(df, S, Nbr):
	"""
	Imputes missing values (-1) using weighted KNN voting.

	Parameters
	----------
	df : pandas.DataFrame
		Encoded dataframe.
	S : numpy.ndarray
		Patient similarity matrix.
	Nbr : numpy.ndarry
		K nearest-neighbor indices
	
	Returns
	---------
	df_imp : pandas.DataFrame
		Completed dataframe.
	"""

	df_imp = df.copy()
	n_patients = len(df_imp)

	for i in range(n_patients):
		# Skip complete patients
		if (-1 not in df_imp.iloc[i].values):
			continue
		# Loop through every features
		for col in df_imp.columns:
			# Only impute missing fields
			if df_imp.at[i, col] != -1:
				continue
			score = defaultdict(float)

			# Loop through KNN
			for j in Nbr[i]:
				if j == -1:
					continue
				value = df_imp.at[j, col]

				# Neighbor also missing
				if value == -1:
					continue

				score[value] += S[i, j]

			# If at least one vote exists
			if len(score) > 0:
				best_value = max(score.items(),
									key = lambda x: x[1])[0]
				df_imp.at[i, col] = best_value
	return df_imp

def find_knn(S, K):
	"""
	Finds the K nearest neighbors for every patient.

	Parameters
	----------
	S : numpy.ndarray
		Asymmetric similarity matrix.
		S[i, j] = similarity of patient j to patient i.
		-1 indicates patient j cannot be compared.
	K : int
		Number of nearest neighbors.

	Returns
	---------
	Nbr : numpy.ndarry
		Shape = (N, K)

		Row i contains the indices of the K nearest neighbors of patient i.
	"""
	N = S.shape[0]
	Nbr = -np.ones((N, K), dtype=int)
	for i in range(N):
		# Comparable patients only
		candidates = []

		for j in range(N):
			if i == j:
				continue
			if S[i, j] == -1:
				continue
			candidates.append((j, S[i, j]))

		# Highest similarity first
		candidates.sort(key=lambda x: x[1], reverse=True)

		# Store neighbor indices
		for t in range(min(K, len(candidates))):
			Nbr[i, t] = candidates[t][0]
	return Nbr

def patient_similarity_matrix(X):
	"""
	Computes an asymmetric patient similarity matrix.

	Parameters
	----------
	X : numpy.ndarray
		N x P feature matrix
		Missing values must be encoded as -1.

	Returns
	----------
	S : numpy.ndarry
		N x N similarity matrix.
		S[i, j] is computed using ONLY the observed features of patient i.
	"""

	N, P = X.shape
	S = -np.ones((N, N))
	for i in range(N):
		# Features available for patient i
		observed = X[i] != -1
		n_features = np.sum(observed)

		if n_features == 0:
			continue

		for j in range(N):
			if i == j:
				S[i, j] = 1.0
				continue
			# Patient j must possess every feature patient i has.
			if np.any((X[j] == -1) & observed):
				continue

			# Count matching values
			matches = np.sum(
				X[i, observed] == X[j, observed]
			)
			# Similarity
			S[i, j] = matches / n_features
	return S

def load_dataframe(excel_file):
	"""
	Reads the patient Excel sheet, removes ID columns, encodes categorical variables, and bins AgeDx.

	Parameters
	----------
	excel_file : str
		Path to Excel file.


	Returns
	----------
	df : pandas.DataFrame
		Encoded dataframe.
	encoders : dict
		Dictionary of categorical encodings.
	"""

	# =====================================
	# Load Excel File
	# =====================================

	df = pd.read_excel(excel_file)

	# =====================================
	# Remove ID Columns
	# =====================================

	id_columns = ["PatientID", "MedRefID"]

	df = df.drop(columns = id_columns, errors="ignore")

	# =====================================
	# Bin Age at Diagnosis
	# =====================================

	if "AgeDx" in df.columns:
		# Convert strings like "049" -> 49
		df["AgeDx"] = pd.to_numeric(df["AgeDx"], errors="coerce")

		def age_bin(age):
			if pd.isna(age):
				return -1
			if age < 25:
				return 0
			elif age < 50:
				return 1
			elif age < 75:
				return 2
			else:
				return 3

		df["AgeDx"] = df["AgeDx"].apply(age_bin)

	# =====================================
	# Encoding Remaining Categorical Values
	# =====================================

	encoders = {}
	for col in df.columns:
		# AgeDx already processed
		if col == "AgeDx":
			continue

		# Skip numeric columns
		if pd.api.types.is_numeric_dtype(df[col]):
			continue
		s = df[col].astype(str).str.strip()

		# Unknown values
		unk_mask = s.str.lower().isin(["unk", "unknown"])

		# Build encoding
		unique_vals = sorted(s[~unk_mask].unique())

		mapping = {v: i for i, v in enumerate(unique_vals)}
		encoded = s.map(mapping)
		encoded[unk_mask] = -1
		df[col] = encoded.astype(int)
		encoders[col] = mapping
	return df, encoders
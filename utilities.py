def get_pssm_spd(data):
  row, col = data.shape
  print (row)
  pssm= data[:, :420]
  spd = data[:, 420:]
  return np.array([pssm[r,:].reshape(21,20) for r in range(row)]), np.array([spd[r, :].reshape(21,8) for r in range(row)])
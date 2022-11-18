def get_pssm_spd(data):
  row, col = data.shape
  print (row)
  pssm= data[:, :620]
  spd = data[:, 620:]
  return np.array([pssm[r,:].reshape(31,20) for r in range(row)]), np.array([spd[r, :].reshape(31,8) for r in range(row)])
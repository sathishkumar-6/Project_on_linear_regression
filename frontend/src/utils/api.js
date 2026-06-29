import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchDatasets = () => api.get('/datasets').then(r => r.data)

export const trainModel = (params) =>
  api.post('/train', params).then(r => r.data)

export const compareLR = (params) =>
  api.get('/compare-lr', { params }).then(r => r.data)

export const predictValue = (params) =>
  api.post('/predict', params).then(r => r.data)

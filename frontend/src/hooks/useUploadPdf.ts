import { useMutation } from '@tanstack/react-query'

import { uploadPdf } from '../api/chat'

export function useUploadPdf() {
  return useMutation({
    mutationFn: async (file: File) => uploadPdf(file),
  })
}

export interface TocEntry {
  id: string
  title: string
}

export const docsToc: TocEntry[] = [
  { id: 'what-is-it', title: 'What is plshelp?' },
  { id: 'install', title: 'Installation' },
  { id: 'quickstart', title: 'Your first library' },
  { id: 'indexing-sites', title: 'Indexing websites' },
  { id: 'indexing-files', title: 'Indexing local files' },
  { id: 'querying', title: 'Querying' },
  { id: 'agent-setup', title: 'Usage with coding agents' },
  { id: 'managing', title: 'Managing libraries' },
  { id: 'debugging', title: 'When results seem wrong' },
  { id: 'pipeline', title: 'The indexing pipeline' },
  { id: 'exports', title: 'Exporting' },
  { id: 'config', title: 'Configuration' },
  { id: 'model-config', title: 'Embedding model config' },
  { id: 'how-search-works', title: 'How search works' },
  { id: 'storage', title: 'Where data lives' },
  { id: 'reference', title: 'Full command reference' },
  { id: 'registry-curl', title: 'Using the registry' },
  { id: 'faq', title: 'FAQ' },
]

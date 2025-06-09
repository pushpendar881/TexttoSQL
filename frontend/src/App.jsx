import React, { useState, useEffect } from 'react';
import { Search, Database, Clock, Loader2, CheckCircle, AlertCircle, Copy, Eye, EyeOff, RefreshCw } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [model, setModel] = useState('mistral-large-latest');
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [showTables, setShowTables] = useState(false);
  const [tables, setTables] = useState(null);
  const [copied, setCopied] = useState(false);
  const [updatingSchema, setUpdatingSchema] = useState(false);


  useEffect(() => {
    fetchStatus();
    fetchModels();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  };

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/schema-tables`);
      const data = await response.json();
      setTables(data);
    } catch (err) {
      console.error('Error fetching tables:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/generate-sql`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          model: model
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate SQL');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const updateSchema = async () => {
    try {
      setUpdatingSchema(true);
      const response = await fetch(`${API_BASE_URL}/update-schema`, {
        method: 'POST',
      });
      
      if (response.ok) {
        await fetchStatus();
        alert('Schema updated successfully!');
      } else {
        const errorData = await response.json();
        alert(`Error updating schema: ${errorData.detail}`);
      }
    } catch (err) {
      alert(`Error updating schema: ${err.message}`);
    } finally {
      setUpdatingSchema(false);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const toggleTables = () => {
    if (!showTables && !tables) {
      fetchTables();
    }
    setShowTables(!showTables);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4 flex items-center justify-center gap-3">
            <Database className="text-blue-600" />
            Natural Language to SQL
          </h1>
          <p className="text-gray-600 text-lg">
            Convert your natural language queries into SQL statements using AI
          </p>
        </div>

        {/* Status Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border border-gray-100">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Database Status
            </h2>
            <button
              onClick={updateSchema}
              disabled={updatingSchema}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {updatingSchema ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Update Schema
            </button>
          </div>
          
          {status && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                {status.db_connected ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-500" />
                )}
                <div>
                  <p className="text-sm text-gray-600">Connection</p>
                  <p className="font-medium">
                    {status.db_connected ? 'Connected' : 'Disconnected'}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                <Database className="w-5 h-5 text-blue-500" />
                <div>
                  <p className="text-sm text-gray-600">Tables</p>
                  <p className="font-medium">{status.schema_tables_count || 0}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                <Clock className="w-5 h-5 text-purple-500" />
                <div>
                  <p className="text-sm text-gray-600">Last Updated</p>
                  <p className="font-medium text-xs">
                    {status.last_updated || 'Never'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Query Form */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border border-gray-100">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Natural Language Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., Show me all customers who made purchases in the last month"
                className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={4}
                required
              />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  AI Model
                </label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="flex items-end">
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
                >
                  {loading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                  {loading ? 'Generating...' : 'Generate SQL'}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <h3 className="font-medium text-red-800">Error</h3>
            </div>
            <p className="text-red-700 mt-2">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* SQL Query Result */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Generated SQL Query</h3>
                <button
                  onClick={() => copyToClipboard(result.sql_query)}
                  className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  <Copy className="w-4 h-4" />
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                  {result.sql_query}
                </pre>
              </div>
              <div className="mt-4 text-sm text-gray-600">
                <p>Execution time: {result.execution_time.toFixed(3)}s</p>
              </div>
            </div>

            {/* Relevant Tables */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Relevant Tables</h3>
              <div className="space-y-3">
                {result.relevant_tables.map((table, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-800">{table.table_name}</h4>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
                        Score: {table.similarity_score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">{table.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Schema Tables */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
          <button
            onClick={toggleTables}
            className="w-full flex items-center justify-between text-lg font-semibold text-gray-800 hover:text-blue-600 transition-colors"
          >
            <span>Database Schema</span>
            {showTables ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
          
          {showTables && tables && (
            <div className="mt-6 space-y-4">
              <p className="text-gray-600">Total tables: {tables.tables_count}</p>
              <div className="grid gap-4">
                {Object.entries(tables.tables).map(([tableName, columns]) => (
                  <div key={tableName} className="border border-gray-200 rounded-lg p-4">
                    <h4 className="font-medium text-gray-800 mb-3">{tableName}</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left p-2 font-medium text-gray-700">Column</th>
                            <th className="text-left p-2 font-medium text-gray-700">Type</th>
                            <th className="text-left p-2 font-medium text-gray-700">Nullable</th>
                          </tr>
                        </thead>
                        <tbody>
                          {columns.map((column, idx) => (
                            <tr key={idx} className="border-b border-gray-100">
                              <td className="p-2 text-gray-800">{column.column_name}</td>
                              <td className="p-2 text-gray-600">{column.data_type}</td>
                              <td className="p-2 text-gray-600">{column.nullable}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
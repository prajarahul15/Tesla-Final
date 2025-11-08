import React, { useState, useRef } from 'react';
import axios from 'axios';
import CitationDisplay from './CitationDisplay';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const FinancialReportAnalyzer = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);
  
  // Q&A State
  const [questions, setQuestions] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [answering, setAnswering] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      setFileId(null);
      setSummary(null);
      setInsights(null);
      setError(null);
      setQuestions([]);
    }
  };

  const handleUpload = async () => {
    if (!uploadedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const uploadResponse = await axios.post(`${API}/fra/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for large files
      });

      if (uploadResponse.data.success) {
        setFileId(uploadResponse.data.file_id);
        
        // Automatically generate summary after upload
        await generateSummary(uploadResponse.data.file_id);
      } else {
        setError(uploadResponse.data.error || 'Failed to upload file');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const generateSummary = async (fId = null) => {
    const id = fId || fileId;
    if (!id) {
      setError('No file uploaded');
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      const response = await axios.post(
        `${API}/fra/summarize`,
        { file_id: id },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 120000,
        }
      );

      if (response.data.success) {
        setSummary(response.data.summary);
        setInsights({
          key_insights: response.data.key_insights || [],
          recommendations: response.data.recommendations || [],
        });
      } else {
        setError(response.data.error || 'Failed to generate summary');
      }
    } catch (err) {
      console.error('Summary error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to generate summary');
    } finally {
      setProcessing(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!currentQuestion.trim() || !fileId) {
      setError('Please enter a question and ensure a file is uploaded');
      return;
    }

    setAnswering(true);
    setError(null);

    const questionText = currentQuestion.trim();

    try {
      const response = await axios.post(
        `${API}/fra/ask`,
        {
          file_id: fileId,
          question: questionText,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 60000,
        }
      );

      if (response.data.success) {
        const newQnA = {
          id: Date.now(),
          question: questionText,
          answer: response.data.answer,
          key_insights: response.data.key_insights || [],
          citations: response.data.citations,
          timestamp: new Date().toISOString(),
        };

        setQuestions((prev) => [...prev, newQnA]);
        setCurrentQuestion('');
      } else {
        setError(response.data.error || 'Failed to get answer');
      }
    } catch (err) {
      console.error('Q&A error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to get answer');
    } finally {
      setAnswering(false);
    }
  };

  const formatText = (text) => {
    if (!text) return '';
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br />');
  };

  return (
    <div className="min-h-screen tesla-gray-light-bg">
      {/* Header */}
      <div className="tesla-white-bg shadow-sm tesla-gray-border border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="tesla-text-3xl tesla-font-bold tesla-black">
            Financial Report Analyzer
          </h1>
          <p className="tesla-text-sm tesla-gray mt-2">
            Upload and analyze financial documents with AI-powered insights
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* File Upload Section */}
        <div className="bg-white rounded-lg shadow-sm tesla-gray-border border p-6 mb-6">
          <h2 className="tesla-text-xl tesla-font-semibold tesla-black mb-4">
            Upload Document
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Supported Formats: PDF, Excel (.xlsx, .xls), CSV, Images (.png, .jpg, etc.)
              </label>
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                accept=".pdf,.xlsx,.xls,.csv,.png,.jpg,.jpeg,.gif,.bmp,.tiff"
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>

            {uploadedFile && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>{uploadedFile.name}</span>
                <span className="text-gray-400">
                  ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!uploadedFile || uploading || processing}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? 'Uploading...' : 'Upload & Process'}
            </button>

            {fileId && (
              <div className="flex items-center space-x-2 text-sm text-green-600">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span>File processed successfully (ID: {fileId.substring(0, 8)}...)</span>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Summary & Insights Section */}
        {(summary || processing) && (
          <div className="bg-white rounded-lg shadow-sm tesla-gray-border border p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="tesla-text-xl tesla-font-semibold tesla-black">
                Document Summary & Insights
              </h2>
              {!processing && fileId && (
                <button
                  onClick={() => generateSummary()}
                  className="text-sm text-blue-600 hover:text-blue-700"
                >
                  Regenerate Summary
                </button>
              )}
            </div>

            {processing ? (
              <div className="flex items-center space-x-2 text-gray-600">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <span>Generating summary and insights...</span>
              </div>
            ) : (
              <>
                {summary && (
                  <div className="mb-6">
                    <h3 className="tesla-text-lg tesla-font-semibold tesla-black mb-2">
                      Executive Summary
                    </h3>
                    <div
                      className="text-gray-700 whitespace-pre-wrap"
                      dangerouslySetInnerHTML={{ __html: formatText(summary) }}
                    />
                  </div>
                )}

                {insights && (
                  <div className="space-y-4">
                    {insights.key_insights && insights.key_insights.length > 0 && (
                      <div>
                        <h3 className="tesla-text-lg tesla-font-semibold tesla-black mb-2">
                          Key Insights
                        </h3>
                        <ul className="list-disc list-inside space-y-1 text-gray-700">
                          {insights.key_insights.map((insight, idx) => (
                            <li key={idx}>
                              {typeof insight === 'string' ? insight : JSON.stringify(insight)}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {insights.recommendations && insights.recommendations.length > 0 && (
                      <div>
                        <h3 className="tesla-text-lg tesla-font-semibold tesla-black mb-2">
                          Recommendations
                        </h3>
                        <ul className="list-disc list-inside space-y-1 text-gray-700">
                          {insights.recommendations.map((rec, idx) => (
                            <li key={idx}>
                              {typeof rec === 'string' ? rec : JSON.stringify(rec)}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Q&A Section */}
        {fileId && (
          <div className="bg-white rounded-lg shadow-sm tesla-gray-border border p-6">
            <h2 className="tesla-text-xl tesla-font-semibold tesla-black mb-4">
              Ask Questions About Your Document
            </h2>

            <div className="space-y-4">
              {/* Question Input */}
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={currentQuestion}
                  onChange={(e) => setCurrentQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !answering && handleAskQuestion()}
                  placeholder="Ask a question about your uploaded document..."
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  disabled={answering}
                />
                <button
                  onClick={handleAskQuestion}
                  disabled={!currentQuestion.trim() || answering}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {answering ? 'Asking...' : 'Ask'}
                </button>
              </div>

              {/* Q&A History */}
              {questions.length > 0 && (
                <div className="space-y-4 mt-6">
                  {questions.map((qna) => (
                    <div key={qna.id} className="border border-gray-200 rounded-lg p-4 space-y-3">
                      {/* Question */}
                      <div className="flex items-start space-x-2">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <p className="font-semibold text-gray-900">{qna.question}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(qna.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>

                      {/* Answer */}
                      <div className="flex items-start space-x-2 ml-10">
                        <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                          <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div
                            className="text-gray-700 whitespace-pre-wrap"
                            dangerouslySetInnerHTML={{ __html: formatText(qna.answer) }}
                          />
                          
                          {/* Citations */}
                          {qna.citations && qna.citations.verified_count > 0 && (
                            <div className="mt-3">
                              <CitationDisplay citationData={qna.citations} />
                            </div>
                          )}

                          {/* Key Insights */}
                          {qna.key_insights && qna.key_insights.length > 0 && (
                            <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                              <h4 className="text-sm font-semibold text-gray-900 mb-2">Key Insights:</h4>
                              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                                {qna.key_insights.map((insight, idx) => (
                                  <li key={idx}>
                                    {typeof insight === 'string' ? insight : JSON.stringify(insight)}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {questions.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <svg className="w-12 h-12 mx-auto mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p>Ask your first question about the uploaded document</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FinancialReportAnalyzer;


// MMM Frontend Application
class MMMApp {
    constructor() {
        this.apiUrl = 'http://mmm-alb-production-190214907.us-east-2.elb.amazonaws.com/api';
        this.uploadId = null;
        this.runId = null;
        this.progressInterval = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const uploadArea = document.getElementById('upload-area');
        const startTrainingBtn = document.getElementById('start-training');
        const cancelTrainingBtn = document.getElementById('cancel-training');

        // File upload events
        browseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files[0]);
        });

        // Training controls
        startTrainingBtn.addEventListener('click', () => this.startTraining());
        cancelTrainingBtn.addEventListener('click', () => this.cancelTraining());
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('upload-status');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
    }

    showSection(sectionId) {
        document.getElementById(sectionId).classList.remove('hidden');
    }

    hideSection(sectionId) {
        document.getElementById(sectionId).classList.add('hidden');
    }

    async handleFileSelect(file) {
        if (!file) return;
        
        if (!file.name.endsWith('.csv')) {
            this.showStatus('❌ Please select a CSV file.', 'error');
            return;
        }

        this.showStatus('📤 Uploading file...', 'info');
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiUrl}/data/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.uploadId = result.upload_id;
                this.showStatus(`✅ File uploaded successfully! Upload ID: ${this.uploadId}`, 'success');
                this.displayDataSummary(result);
                this.displayChannelInfo(result.channel_info);
                this.showSection('summary-section');
                this.showSection('config-section');
                document.getElementById('start-training').disabled = false;
            } else {
                this.showStatus(`❌ Upload failed: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.showStatus(`❌ Upload error: ${error.message}`, 'error');
        }
    }

    displayDataSummary(data) {
        const summary = data.data_summary;
        const summaryHtml = `
            <div class="results-grid">
                <div class="metric-card">
                    <div class="metric-value">${summary.total_days}</div>
                    <div class="metric-label">Total Days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">$${this.formatNumber(summary.total_profit)}</div>
                    <div class="metric-label">Total Profit</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">$${this.formatNumber(summary.total_annual_spend)}</div>
                    <div class="metric-label">Annual Spend</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.channel_count}</div>
                    <div class="metric-label">Marketing Channels</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.data_quality_score}%</div>
                    <div class="metric-label">Data Quality Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.business_tier}</div>
                    <div class="metric-label">Business Tier</div>
                </div>
            </div>
            <p style="margin-top: 20px; color: #666;">
                📅 Date Range: ${new Date(summary.date_range.start).toLocaleDateString()} - ${new Date(summary.date_range.end).toLocaleDateString()}
            </p>
        `;
        document.getElementById('data-summary').innerHTML = summaryHtml;
    }

    displayChannelInfo(channels) {
        const channelHtml = Object.entries(channels).map(([name, info]) => `
            <div class="channel-card">
                <div class="channel-name">${this.formatChannelName(name)}</div>
                <div class="channel-stats">
                    <div>Type: ${info.type}</div>
                    <div>Spend: $${this.formatNumber(info.total_spend)}</div>
                    <div>Share: ${(info.spend_share * 100).toFixed(1)}%</div>
                    <div>Active Days: ${info.days_active}</div>
                </div>
            </div>
        `).join('');
        
        document.getElementById('channel-info').innerHTML = `
            <h3 style="margin-top: 30px; margin-bottom: 15px;">📺 Marketing Channels</h3>
            <div class="channel-info">${channelHtml}</div>
        `;
    }

    async startTraining() {
        if (!this.uploadId) {
            this.showStatus('❌ No data uploaded', 'error');
            return;
        }

        const config = {
            carryover_prior: document.getElementById('carryover-prior').value,
            saturation_prior: document.getElementById('saturation-prior').value,
            media_transform: document.getElementById('media-transform').value,
            max_lag: parseInt(document.getElementById('max-lag').value),
            iterations: parseInt(document.getElementById('iterations').value)
        };

        try {
            document.getElementById('start-training').disabled = true;
            this.showSection('progress-section');
            
            const response = await fetch(`${this.apiUrl}/model/train?upload_id=${this.uploadId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (response.ok) {
                this.runId = result.run_id;
                this.showTrainingStatus(`🚀 Training started! Run ID: ${this.runId}`, 'success');
                document.getElementById('cancel-training').style.display = 'block'; // Show cancel button
                this.startProgressMonitoring();
            } else {
                this.showTrainingStatus(`❌ Training failed to start: ${result.detail}`, 'error');
                document.getElementById('start-training').disabled = false;
            }
        } catch (error) {
            this.showTrainingStatus(`❌ Training error: ${error.message}`, 'error');
            document.getElementById('start-training').disabled = false;
        }
    }

    async cancelTraining() {
        if (!this.runId) {
            this.showTrainingStatus('❌ No active training to cancel', 'error');
            return;
        }

        const confirmCancel = confirm('Are you sure you want to cancel the training? This action cannot be undone.');
        if (!confirmCancel) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/model/training/cancel/${this.runId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (response.ok) {
                this.showTrainingStatus('⏹️ Training cancelled successfully', 'error');
                this.stopProgressMonitoring();
                document.getElementById('cancel-training').style.display = 'none';
                document.getElementById('start-training').disabled = false;
            } else {
                this.showTrainingStatus(`❌ Failed to cancel training: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.showTrainingStatus(`❌ Cancel error: ${error.message}`, 'error');
        }
    }

    showTrainingStatus(message, type) {
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
    }

    startProgressMonitoring() {
        this.progressInterval = setInterval(() => {
            this.checkTrainingProgress();
        }, 3000); // Check every 3 seconds
        
        // Also check immediately
        this.checkTrainingProgress();
    }

    async checkTrainingProgress() {
        if (!this.runId) return;

        try {
            const response = await fetch(`${this.apiUrl}/model/training/progress/${this.runId}`);
            const progress = await response.json();

            if (response.ok) {
                this.updateProgressDisplay(progress);
                
                if (progress.status === 'completed') {
                    this.stopProgressMonitoring();
                    this.showTrainingComplete(progress);
                } else if (progress.status === 'failed') {
                    this.stopProgressMonitoring();
                    this.showTrainingFailed(progress);
                } else if (progress.status === 'cancelled') {
                    this.stopProgressMonitoring();
                    this.showTrainingCancelled(progress);
                }
            } else {
                console.error('Progress check failed:', progress);
            }
        } catch (error) {
            console.error('Progress monitoring error:', error);
        }
    }

    updateProgressDisplay(progress) {
        const progressFill = document.getElementById('progress-fill');
        const progressPct = progress.progress?.progress_pct || 0;
        
        progressFill.style.width = `${Math.min(progressPct, 100)}%`;
        
        let statusMessage = `Status: ${progress.status}`;
        if (progress.progress?.current_step) {
            statusMessage += ` - ${progress.progress.current_step}`;
        }
        if (progress.progress?.current_fold && progress.progress?.total_folds) {
            statusMessage += ` (${progress.progress.current_fold}/${progress.progress.total_folds})`;
        }
        
        this.showTrainingStatus(statusMessage, 'info');
    }

    showTrainingComplete(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '100%';
        this.showTrainingStatus('🎉 Training completed successfully!', 'success');
        document.getElementById('cancel-training').style.display = 'none';
        
        // Show results if available
        if (progress.results) {
            this.displayResults(progress.results);
        }
    }

    showTrainingFailed(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        const errorMsg = progress.error || 'Training failed';
        this.showTrainingStatus(`❌ Training failed: ${errorMsg}`, 'error');
        document.getElementById('cancel-training').style.display = 'none';
        document.getElementById('start-training').disabled = false;
    }

    showTrainingCancelled(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        const cancelMsg = progress.progress?.message || 'Training cancelled by user';
        this.showTrainingStatus(`⏹️ ${cancelMsg}`, 'error');
        document.getElementById('cancel-training').style.display = 'none';
        document.getElementById('start-training').disabled = false;
    }

    displayResults(results) {
        this.showSection('results-section');
        
        const resultsHtml = `
            <div class="metric-card">
                <div class="metric-value">${results.cv_mape?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">CV MAPE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${results.r_squared?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">R-Squared</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${results.final_mape?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">Final MAPE</div>
            </div>
        `;
        
        document.getElementById('results-grid').innerHTML = resultsHtml;
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toFixed(0);
    }

    formatChannelName(name) {
        return name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MMMApp();
});

// Handle page unload to clean up intervals
window.addEventListener('beforeunload', () => {
    if (window.mmmApp && window.mmmApp.progressInterval) {
        clearInterval(window.mmmApp.progressInterval);
    }
});
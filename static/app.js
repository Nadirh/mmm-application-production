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
            carryover_prior: "uniform",
            saturation_prior: "uniform", 
            media_transform: "adstock",
            max_lag: 8,
            iterations: 2000
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
        const confirmCancel = confirm('Are you sure you want to cancel the training? This action cannot be undone.');
        if (!confirmCancel) {
            return;
        }

        let runIdToCancel = this.runId;

        // If we don't have a runId in memory, try to find an active training job
        if (!runIdToCancel) {
            try {
                this.showTrainingStatus('🔍 Looking for active training jobs...', 'info');
                const response = await fetch(`${this.apiUrl}/admin/training/list`);
                const result = await response.json();
                if (response.ok && result.training_runs && result.training_runs.length > 0) {
                    // Find the most recent training or queued job
                    const activeJob = result.training_runs.find(run => 
                        run.status === 'training' || run.status === 'queued'
                    );
                    if (activeJob) {
                        runIdToCancel = activeJob.run_id;
                        this.runId = runIdToCancel; // Update our local state
                        this.showTrainingStatus(`🔍 Found stuck training job: ${runIdToCancel.substring(0, 8)}...`, 'info');
                    } else {
                        this.showTrainingStatus('❌ No active training to cancel', 'error');
                        return;
                    }
                } else {
                    this.showTrainingStatus('❌ No active training to cancel', 'error');
                    return;
                }
            } catch (error) {
                this.showTrainingStatus(`❌ Error checking for active training: ${error.message}`, 'error');
                return;
            }
        }

        try {
            const response = await fetch(`${this.apiUrl}/model/training/cancel/${runIdToCancel}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (response.ok) {
                this.showTrainingStatus('⏹️ Training cancelled successfully', 'success');
                this.stopProgressMonitoring();
                document.getElementById('cancel-training').style.display = 'none';
                document.getElementById('start-training').disabled = false;
                this.runId = null; // Clear the runId
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

    async showTrainingComplete(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '100%';
        this.showTrainingStatus('🎉 Training completed successfully!', 'success');
        document.getElementById('cancel-training').style.display = 'none';
        
        // Fetch and show full results
        if (this.runId) {
            try {
                this.showTrainingStatus('🎉 Training completed! Fetching results...', 'success');
                const response = await fetch(`${this.apiUrl}/model/results/${this.runId}`);
                const results = await response.json();
                
                if (response.ok) {
                    this.displayResults(results);
                    // Fetch marginal ROI data separately
                    await this.fetchAndDisplayMarginalROI();
                } else {
                    console.error('Failed to fetch results:', results);
                    this.showTrainingStatus('🎉 Training completed! (Results unavailable)', 'success');
                }
            } catch (error) {
                console.error('Error fetching results:', error);
                this.showTrainingStatus('🎉 Training completed! (Results unavailable)', 'success');
            }
        } else {
            // Fallback: try to show results from progress if available
            if (progress.results) {
                this.displayResults(progress.results);
            }
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
        
        // Handle both old format (progress.results) and new format (API response)
        const performance = results.model_performance || results;
        const parameters = results.parameters;
        
        const resultsHtml = `
            <div class="metric-card">
                <div class="metric-value">${performance.cv_mape?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">CV MAPE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${performance.r_squared?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">R-Squared</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(performance.mape || performance.final_mape)?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">Final MAPE</div>
            </div>
        `;
        
        document.getElementById('results-grid').innerHTML = resultsHtml;
        
        // Display parameter values if available
        if (parameters && (parameters.channel_alphas || parameters.channel_betas || parameters.channel_rs)) {
            this.displayParameters(parameters);
            this.displayResponseCurveEquations(parameters);
        }
        
        // Display marginal ROI if available
        if (results.marginal_roi && results.marginal_roi.marginal_roi_by_channel) {
            this.displayMarginalROI(results.marginal_roi);
        }
    }

    displayParameters(parameters) {
        const { channel_alphas, channel_betas, channel_rs } = parameters;
        
        if (!channel_alphas || !channel_betas || !channel_rs) {
            return;
        }
        
        // Create parameter display section
        const parameterHtml = Object.keys(channel_alphas).map(channel => {
            const alpha = channel_alphas[channel]?.toFixed(3) || 'N/A';
            const beta = channel_betas[channel]?.toFixed(3) || 'N/A';
            const r = channel_rs[channel]?.toFixed(3) || 'N/A';
            
            return `
                <div class="parameter-card">
                    <div class="parameter-channel">${this.formatChannelName(channel)}</div>
                    <div class="parameter-values">
                        <div class="parameter-item">
                            <span class="parameter-label">Alpha (Strength):</span>
                            <span class="parameter-value">${alpha}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="parameter-label">Beta (Saturation):</span>
                            <span class="parameter-value">${beta}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="parameter-label">R (Adstock):</span>
                            <span class="parameter-value">${r}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add parameters section to results
        const parametersSection = `
            <div style="margin-top: 30px;">
                <h3 style="color: #333; margin-bottom: 20px;">📊 Optimized Parameters</h3>
                <div class="parameters-grid">
                    ${parameterHtml}
                </div>
            </div>
        `;
        
        document.getElementById('results-grid').insertAdjacentHTML('afterend', parametersSection);
    }

    displayResponseCurveEquations(parameters) {
        const { channel_alphas, channel_betas, channel_rs } = parameters;

        if (!channel_alphas || !channel_betas || !channel_rs) {
            return;
        }

        // Create equation display section
        const equationHtml = Object.keys(channel_alphas).map(channel => {
            const alpha = channel_alphas[channel];
            const beta = channel_betas[channel];
            const r = channel_rs[channel];

            return `
                <div class="equation-card">
                    <div class="equation-channel">${this.formatChannelName(channel)}</div>
                    <div class="equation-formula">
                        <div class="equation-text">
                            Profit = α × [(S / (1 - r))^β]
                        </div>
                        <div class="equation-explanation">
                            Where S is daily spend. Adstock transforms spend as S/(1-r), then saturation is applied with power β
                        </div>
                    </div>
                    <div class="equation-parameters">
                        <div class="parameter-row">
                            <span class="parameter-symbol">α</span>
                            <span class="parameter-description">Incremental strength</span>
                            <span class="parameter-value">${alpha.toFixed(4)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">β</span>
                            <span class="parameter-description">Saturation (diminishing returns)</span>
                            <span class="parameter-value">${beta.toFixed(3)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">r</span>
                            <span class="parameter-description">Adstock (carryover effect)</span>
                            <span class="parameter-value">${r.toFixed(3)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add equations section to results
        const equationsSection = `
            <div style="margin-top: 30px;">
                <h3 style="color: #333; margin-bottom: 20px;">📈 Response Curve Equations</h3>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #666;">
                    These equations show how incremental profit responds to spending on each channel, accounting for adstock (carryover effects) and saturation (diminishing returns).
                </div>
                <div class="equations-grid">
                    ${equationHtml}
                </div>
            </div>
        `;

        // Insert after the parameters section
        const parametersGrid = document.querySelector('.parameters-grid');
        if (parametersGrid && parametersGrid.parentElement) {
            parametersGrid.parentElement.insertAdjacentHTML('afterend', equationsSection);
        } else {
            document.getElementById('results-grid').insertAdjacentHTML('afterend', equationsSection);
        }
    }

    displayMarginalROI(marginalROIData) {
        const { marginal_roi_by_channel, current_spend, baseline_spend_per_day, interpretation } = marginalROIData;
        
        if (!marginal_roi_by_channel || Object.keys(marginal_roi_by_channel).length === 0) {
            return;
        }
        
        // Create marginal ROI display section
        const marginalROIHtml = Object.entries(marginal_roi_by_channel).map(([channel, roi]) => {
            const currentSpendValue = current_spend[channel] || 0;
            const baselineSpendValue = baseline_spend_per_day[channel] || 0;
            const formattedROI = roi.toFixed(2);
            
            return `
                <div class="marginal-roi-card">
                    <div class="roi-channel">${this.formatChannelName(channel)}</div>
                    <div class="roi-values">
                        <div class="roi-item">
                            <span class="roi-label">Marginal ROI:</span>
                            <span class="roi-value">$${formattedROI}</span>
                        </div>
                        <div class="roi-item">
                            <span class="roi-label">Baseline Spend (30-day avg):</span>
                            <span class="roi-value">$${this.formatNumber(baselineSpendValue)}/day</span>
                        </div>
                        ${currentSpendValue > 0 ? `
                        <div class="roi-item">
                            <span class="roi-label">Current Daily Spend:</span>
                            <span class="roi-value">$${this.formatNumber(currentSpendValue)}</span>
                        </div>
                        ` : ''}
                        <div class="roi-interpretation">
                            Each additional $1/day generates $${formattedROI} more profit
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add marginal ROI section to results
        const marginalROISection = `
            <div style="margin-top: 30px;">
                <h3 style="color: #333; margin-bottom: 20px;">💰 Marginal ROI by Channel</h3>
                <div class="interpretation-box" style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #666;">
                    ${interpretation.description} (${interpretation.units})
                    <br><strong>Channels ranked by efficiency - highest ROI first</strong>
                    ${interpretation.baseline_note ? `<br><br><em>${interpretation.baseline_note}</em>` : ''}
                </div>
                <div class="marginal-roi-grid">
                    ${marginalROIHtml}
                </div>
            </div>
        `;
        
        // Insert after the parameters section if it exists, otherwise after the results grid
        const parametersGrid = document.querySelector('.parameters-grid');
        if (parametersGrid && parametersGrid.parentElement) {
            parametersGrid.parentElement.insertAdjacentHTML('afterend', marginalROISection);
        } else {
            document.getElementById('results-grid').insertAdjacentHTML('afterend', marginalROISection);
        }
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    async fetchAndDisplayMarginalROI() {
        try {
            const response = await fetch(`${this.apiUrl}/model/marginal-roi/${this.runId}`);
            const marginalROIData = await response.json();
            
            if (response.ok && marginalROIData.marginal_roi_by_channel) {
                this.displayMarginalROI(marginalROIData);
            } else {
                console.warn('Marginal ROI data not available:', marginalROIData);
            }
        } catch (error) {
            console.warn('Error fetching marginal ROI:', error);
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
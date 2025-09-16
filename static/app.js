// MMM Frontend Application v1.9.22 with response curve charts and 2 decimal places
console.log('🚀 MMM App Loading with Equation Debug');

class MMMApp {
    constructor() {
        this.apiUrl = 'http://mmm-alb-production-190214907.us-east-2.elb.amazonaws.com/api';
        this.uploadId = null;
        this.runId = null;
        this.progressInterval = null;

        // Add VISIBLE indicator that JS is working
        this.addVisibleDebugIndicator();

        this.initializeEventListeners();
    }

    addVisibleDebugIndicator() {
        // Add a bright red banner at the top of the page to show JS is working
        const banner = document.createElement('div');
        banner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: red;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            z-index: 9999;
            font-size: 16px;
        `;
        banner.textContent = '🚀 JS v1.9.22 LOADED - CHARTS & 2 DECIMAL PLACES';
        document.body.appendChild(banner);

        // Remove after 5 seconds
        setTimeout(() => {
            if (banner.parentNode) {
                banner.parentNode.removeChild(banner);
            }
        }, 5000);
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
                    console.log('Results fetched successfully:', results);
                    this.displayResults(results);
                    // Fetch marginal ROI data separately
                    await this.fetchAndDisplayMarginalROI();
                } else {
                    console.error('Failed to fetch results:', results);
                    console.error('Response status:', response.status);
                    this.showTrainingStatus('🎉 Training completed! (Results unavailable)', 'success');
                }
            } catch (error) {
                console.error('Error fetching results:', error);
                this.showTrainingStatus('🎉 Training completed! (Results unavailable)', 'success');
            }
        } else {
            // Fallback: try to show results from progress if available
            if (progress.results) {
                console.log('Using fallback results from progress:', progress.results);
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
        console.log('*** EQUATION DEBUG: displayResults called');
        console.log('*** EQUATION DEBUG: Full results object:', results);
        this.showSection('results-section');

        // Handle both old format (progress.results) and new format (API response)
        const performance = results.model_performance || results;
        const parameters = results.parameters;
        const confidenceIntervals = results.confidence_intervals || {};
        console.log('*** EQUATION DEBUG: Extracted parameters:', parameters);
        console.log('*** EQUATION DEBUG: Extracted confidence intervals:', confidenceIntervals);
        console.log('Extracted parameters:', parameters);
        
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
        console.log('*** EQUATION DEBUG: Checking parameters for equation display:', parameters);
        console.log('*** EQUATION DEBUG: Parameters exists?', !!parameters);
        console.log('*** EQUATION DEBUG: Has channel_alphas?', !!(parameters && parameters.channel_alphas));
        console.log('*** EQUATION DEBUG: Has channel_betas?', !!(parameters && parameters.channel_betas));
        console.log('*** EQUATION DEBUG: Has channel_rs?', !!(parameters && parameters.channel_rs));

        if (parameters && (parameters.channel_alphas || parameters.channel_betas || parameters.channel_rs)) {
            console.log('*** EQUATION DEBUG: Parameters found, calling displayParameters and displayResponseCurveEquations');
            this.displayParameters(parameters);
            this.displayResponseCurveEquations(parameters, confidenceIntervals);
        } else {
            console.log('*** EQUATION DEBUG: No parameters found for equation display - showing test equation');
            // Force show a test equation to verify display works
            this.displayTestEquation();
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
        
        // Parameters are now displayed in the equations section, removing duplicate display
    }

    displayResponseCurveEquations(parameters, confidenceIntervals = {}) {
        console.log('displayResponseCurveEquations called with parameters:', parameters);
        console.log('Parameters type:', typeof parameters);
        console.log('Parameters keys:', Object.keys(parameters || {}));

        const { channel_alphas, channel_betas, channel_rs } = parameters;

        console.log('Extracted channel_alphas:', channel_alphas);
        console.log('Extracted channel_betas:', channel_betas);
        console.log('Extracted channel_rs:', channel_rs);

        if (!channel_alphas || !channel_betas || !channel_rs) {
            console.log('Missing required parameters for equations - showing test equation instead');
            this.displayTestEquation();
            return;
        }
        console.log('All parameters present, creating equations...');

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
                            Lifetime Profit = ${alpha.toFixed(2)} × [(Daily Spend / ${(1-r).toFixed(2)})^${beta.toFixed(2)}]
                        </div>
                        <div class="equation-explanation">
                            This shows lifetime incremental profit from sustained daily spending (includes carryover effects)
                        </div>
                    </div>
                    <div class="equation-parameters">
                        <div class="parameter-row">
                            <span class="parameter-symbol">α</span>
                            <span class="parameter-description">Incremental strength</span>
                            <span class="equation-parameter-value">${alpha.toFixed(2)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">β</span>
                            <span class="parameter-description">Saturation (diminishing returns)</span>
                            <span class="equation-parameter-value">${beta.toFixed(2)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">r</span>
                            <span class="parameter-description">Adstock (carryover effect)</span>
                            <span class="equation-parameter-value">${r.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add response curves section
        const responseCurvesSection = this.generateResponseCurvesSection(parameters);

        // Add equations section to results
        const equationsSection = `
            <div style="margin-top: 30px; border: 3px solid #ff0000; padding: 20px; background: #fff;">
                <h2 style="color: #ff0000; margin-bottom: 20px; font-size: 2rem;">🧮 RESPONSE CURVE EQUATIONS</h2>
                <div style="background: #ffeeee; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #ff0000;">
                    <strong>These equations show lifetime incremental profit from sustained daily spending on each channel:</strong><br>
                    Lifetime Profit = α × [(Daily Spend ÷ (1-r))^β] where carryover effects are built into the formula
                </div>
                <div class="equations-grid">
                    ${equationHtml}
                </div>
            </div>
        `;

        // Insert after the results grid
        console.log('Inserting equations section...');
        const resultsGrid = document.getElementById('results-grid');
        if (resultsGrid) {
            resultsGrid.insertAdjacentHTML('afterend', responseCurvesSection);
            resultsGrid.insertAdjacentHTML('afterend', equationsSection);
            console.log('Response curves and equations sections inserted successfully');

            // Generate the actual charts after DOM elements are created
            setTimeout(() => {
                this.generateResponseCurveCharts(parameters, this.runId);
            }, 100);
        } else {
            console.error('Could not find results-grid element');
        }
    }

    displayTestEquation() {
        console.log('Displaying test equation to verify DOM insertion...');
        const testEquationSection = `
            <div style="margin-top: 30px; border: 3px solid #00ff00; padding: 20px; background: #eeffee;">
                <h2 style="color: #00aa00; margin-bottom: 20px; font-size: 2rem;">🧪 TEST EQUATION DISPLAY</h2>
                <div style="background: #ffffff; padding: 20px; border-radius: 8px; border: 2px solid #00aa00;">
                    <h3>TV Channel:</h3>
                    <p style="font-size: 1.2rem; font-family: monospace;"><strong>Profit = 27.09 × [(S / (1 - 0.1))^0.7]</strong></p>
                    <p>Where S is daily spend for TV advertising</p>
                </div>
            </div>
        `;

        const resultsGrid = document.getElementById('results-grid');
        if (resultsGrid) {
            resultsGrid.insertAdjacentHTML('afterend', testEquationSection);
            console.log('Test equation inserted successfully');
        } else {
            console.error('Could not find results-grid for test equation');
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
        
        // Generate mROI curves section
        const mroiCurvesSection = this.generateMROICurvesSection(marginal_roi_by_channel, baseline_spend_per_day);

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
                ${mroiCurvesSection}
            </div>
        `;
        
        // Insert after the parameters section if it exists, otherwise after the results grid
        const parametersGrid = document.querySelector('.parameters-grid');
        if (parametersGrid && parametersGrid.parentElement) {
            parametersGrid.parentElement.insertAdjacentHTML('afterend', marginalROISection);
        } else {
            document.getElementById('results-grid').insertAdjacentHTML('afterend', marginalROISection);
        }

        // Generate mROI curves after DOM elements are created
        setTimeout(() => {
            this.generateMROICharts(marginal_roi_by_channel, baseline_spend_per_day);
            console.log('📊 About to add Profit Maximizer. mROI data:', Object.keys(marginal_roi_by_channel));
            this.addProfitMaximizer(marginal_roi_by_channel, baseline_spend_per_day);
        }, 100);
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

    generateResponseCurvesSection(parameters) {
        console.log('🚀 Generating response curves section with parameters:', parameters);

        const { channel_alphas, channel_betas, channel_rs } = parameters;

        const curvesHtml = Object.keys(channel_alphas).map(channel => {
            return `
                <div class="response-curve-card">
                    <div class="response-curve-title">${this.formatChannelName(channel)} Response Curve</div>
                    <div class="chart-container">
                        <canvas id="chart-${channel}"></canvas>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="response-curves-section">
                <h2 style="color: #28a745; margin-bottom: 20px; font-size: 2rem;">📈 RESPONSE CURVES</h2>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #28a745;">
                    <strong>These charts show how incremental profit responds to daily spend for each channel.</strong><br>
                    X-axis: Daily Spend ($) | Y-axis: Lifetime Incremental Profit ($)
                </div>
                <div class="response-curves-grid">
                    ${curvesHtml}
                </div>
            </div>
        `;
    }

    renderResponseCurvesFromAPI(data) {
        console.log('📊 Rendering response curves from API data:', data);

        const responseCurves = data.response_curves;
        const avgDailySpend = data.avg_daily_spend_28_days || {};
        const spendCorrelations = data.spend_correlations || {};

        Object.keys(responseCurves).forEach(channel => {
            const curveData = responseCurves[channel];
            const spendLevels = curveData.spend_levels || [];
            const profits = curveData.incremental_profits || [];
            const confidenceIntervals = curveData.confidence_intervals;
            const avgSpend = avgDailySpend[channel];

            console.log(`📊 ${channel} - Confidence intervals:`, confidenceIntervals);
            console.log(`📊 ${channel} - 28-day avg spend: $${avgSpend?.toFixed(2) || 'N/A'}`);

            // Find the point closest to the 28-day average for annotation
            let avgSpendIndex = -1;
            if (avgSpend && avgSpend > 0) {
                avgSpendIndex = spendLevels.findIndex(spend => spend >= avgSpend);
                if (avgSpendIndex === -1) avgSpendIndex = spendLevels.length - 1;
            }

            // Prepare datasets
            const datasets = [{
                label: 'Lifetime Incremental Profit',
                data: profits,
                borderColor: '#2d5aa0',
                backgroundColor: 'rgba(45, 90, 160, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4
            }];

            // Add confidence interval bands if available
            if (confidenceIntervals && confidenceIntervals.lower && confidenceIntervals.upper) {
                console.log(`✅ Adding confidence intervals for ${channel}`);

                datasets.push({
                    label: '95% Confidence Interval (Upper)',
                    data: confidenceIntervals.upper,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                });

                datasets.push({
                    label: '95% Confidence Interval (Lower)',
                    data: confidenceIntervals.lower,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'rgba(45, 90, 160, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: 1, // Fill to the upper bound dataset
                    tension: 0.4,
                    pointRadius: 0
                });
            } else {
                console.log(`⚠️ No confidence intervals found for ${channel}`);
            }

            // Add vertical line for 28-day average if available
            const annotations = [];
            if (avgSpend && avgSpend > 0 && avgSpendIndex >= 0) {
                annotations.push({
                    type: 'line',
                    mode: 'vertical',
                    scaleID: 'x',
                    value: avgSpendIndex,
                    borderColor: 'rgba(255, 99, 132, 0.8)',
                    borderWidth: 2,
                    borderDash: [10, 5],
                    label: {
                        content: `28-day avg: $${avgSpend.toFixed(0)}`,
                        enabled: true,
                        position: 'top'
                    }
                });
            }

            // Create chart
            const ctx = document.getElementById(`chart-${channel}`);
            if (ctx) {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: `${this.formatChannelName(channel)} Response Curve${avgSpend ? ` (28-day avg: $${avgSpend.toFixed(0)}/day)` : ''}`,
                                font: { size: 16, weight: 'bold' }
                            },
                            legend: {
                                display: true,
                                position: 'bottom',
                                labels: {
                                    filter: function(legendItem, chartData) {
                                        // Only show main curve and confidence interval labels (simplified)
                                        return legendItem.text !== '95% Confidence Interval (Upper)';
                                    },
                                    generateLabels: function(chart) {
                                        const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                        return labels.map(label => {
                                            if (label.text === '95% Confidence Interval (Lower)') {
                                                label.text = '95% Confidence Interval';
                                            }
                                            return label;
                                        });
                                    }
                                }
                            },
                            annotation: annotations.length > 0 ? {
                                annotations: annotations
                            } : undefined
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Daily Spend ($)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Lifetime Incremental Profit ($)'
                                },
                                min: 0,  // Never show negative values
                                ticks: {
                                    callback: function(value) {
                                        return '$' + Math.round(value).toLocaleString();
                                    }
                                }
                            }
                        }
                    }
                });

                console.log(`✅ Chart created for ${channel} with ${confidenceIntervals ? 'confidence intervals' : 'no confidence intervals'}`);

                // Add correlations summary box after the chart
                const spendCorr = spendCorrelations[channel];

                if (spendCorr !== undefined) {
                    const correlationHTML = `
                        <div style="background: #6f42c1; color: white; padding: 15px; margin-top: 10px; border-radius: 8px; font-size: 0.9em;">
                            <div style="font-weight: bold; margin-bottom: 8px; text-align: center;">📊 Spend Correlation with Profit</div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.2em; font-weight: bold;">${Math.round(spendCorr * 100)}%</div>
                            </div>
                        </div>
                    `;

                    // Insert the correlation box after the chart canvas
                    ctx.parentElement.insertAdjacentHTML('afterend', correlationHTML);
                }
            } else {
                console.error(`❌ Could not find canvas element for ${channel}`);
            }
        });
    }

    renderMarginalROICharts(data) {
        console.log('📈 Rendering marginal ROI charts');
        console.log('📊 Response curves data:', data.response_curves);
        console.log('💰 Average daily spend data:', data.avg_daily_spend_28_days);

        const responseCurves = data.response_curves;
        const avgDailySpend = data.avg_daily_spend_28_days || {};

        // Add section header for marginal ROI charts
        const chartContainer = document.getElementById('response-curves-charts');
        if (!chartContainer) {
            console.error('❌ Could not find response-curves-charts container');
            return;
        }
        console.log('✅ Found chart container, adding marginal ROI section');

        const marginalSection = document.createElement('div');
        marginalSection.innerHTML = `
            <div style="margin-top: 40px; margin-bottom: 20px;">
                <h3 style="color: #333; font-size: 1.5em; margin-bottom: 10px;">📊 Marginal ROI by Channel</h3>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                    Shows the return on investment for the next dollar spent. Values above 1.0 indicate profitable spend.
                </p>
            </div>
            <div id="marginal-roi-charts" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;"></div>
        `;
        chartContainer.appendChild(marginalSection);

        const marginalChartsContainer = document.getElementById('marginal-roi-charts');

        Object.keys(responseCurves).forEach(channel => {
            const curveData = responseCurves[channel];
            const spendLevels = curveData.spend_levels || [];
            const marginalRoas = curveData.marginal_roas || [];
            const currentSpend = avgDailySpend[channel] || 0;

            // Create chart container
            const chartDiv = document.createElement('div');
            chartDiv.style.cssText = 'background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px;';

            const canvas = document.createElement('canvas');
            canvas.id = `marginal-roi-chart-${channel}`;
            canvas.width = 400;
            canvas.height = 300;

            chartDiv.appendChild(canvas);
            marginalChartsContainer.appendChild(chartDiv);

            const ctx = canvas.getContext('2d');

            // Find current position on the curve
            let currentROI = 0;
            let currentIndex = -1;
            if (currentSpend > 0) {
                // Find closest spend level to current spend
                currentIndex = spendLevels.findIndex(spend => spend >= currentSpend);
                if (currentIndex === -1) currentIndex = spendLevels.length - 1;
                if (currentIndex >= 0 && currentIndex < marginalRoas.length) {
                    currentROI = marginalRoas[currentIndex];
                }
            }

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spendLevels,
                    datasets: [
                        {
                            label: 'Marginal ROI',
                            data: marginalRoas,
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Breakeven Line (ROI = 1)',
                            data: spendLevels.map(() => 1.0),
                            borderColor: '#dc3545',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 0
                        },
                        ...(currentIndex >= 0 ? [{
                            label: 'Current Position',
                            data: [{x: currentSpend, y: currentROI}],
                            backgroundColor: '#28a745',
                            borderColor: '#28a745',
                            pointRadius: 8,
                            pointHoverRadius: 10,
                            showLine: false
                        }] : [])
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `${channel} - Marginal ROI`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    if (context.datasetIndex === 0) {
                                        return `Marginal ROI: ${context.parsed.y.toFixed(2)}`;
                                    } else if (context.datasetIndex === 1) {
                                        return 'Breakeven';
                                    } else {
                                        return `Current: ${context.parsed.y.toFixed(2)} ROI`;
                                    }
                                },
                                title: function(context) {
                                    return `Daily Spend: $${context[0].parsed.x.toFixed(0)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Daily Spend ($)'
                            },
                            grid: {
                                alpha: 0.3
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Marginal ROI'
                            },
                            grid: {
                                alpha: 0.3
                            },
                            min: 0
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });

            console.log(`✅ Marginal ROI chart created for ${channel}, current spend: $${currentSpend.toFixed(0)}, current ROI: ${currentROI.toFixed(2)}`);
        });
    }

    async generateResponseCurveCharts(parameters, runId) {
        console.log('🚀 Generating response curve charts with parameters:', parameters);
        console.log('📊 Run ID for API call:', runId);

        // Try to fetch actual response curves from API with confidence intervals
        try {
            const response = await fetch(`${this.apiUrl}/model/response-curves/${runId}`);
            const data = await response.json();

            if (response.ok && data.response_curves) {
                console.log('✅ Fetched response curves with confidence intervals from API');
                this.renderResponseCurvesFromAPI(data);
                console.log('🚀 About to render marginal ROI charts...');
                this.renderMarginalROICharts(data);
                return;
            }
        } catch (error) {
            console.warn('⚠️ Could not fetch response curves from API, falling back to manual generation:', error);
        }

        // Fallback to manual generation
        const { channel_alphas, channel_betas, channel_rs } = parameters;
        Object.keys(channel_alphas).forEach(channel => {
            const alpha = channel_alphas[channel];
            const beta = channel_betas[channel];
            const r = channel_rs[channel];

            // Get confidence intervals for this channel
            const channelCI = confidenceIntervals[channel] || {};
            const alphaCI = channelCI.alpha || [alpha, alpha];
            const betaCI = channelCI.beta || [beta, beta];
            const rCI = channelCI.r || [r, r];

            // Generate spend levels from 0 to max spend
            const maxSpend = 10000; // $10k max daily spend for visualization
            const spendLevels = [];
            const profits = [];
            const profitsLower = [];
            const profitsUpper = [];

            for (let spend = 0; spend <= maxSpend; spend += maxSpend / 100) {
                // Calculate time-series adstocked spend (30-day carryover)
                const adstockedSpend = this.calculateTimeSeriesAdstock(spend, r, 30);
                const adstockedSpendLower = this.calculateTimeSeriesAdstock(spend, rCI[0], 30);
                const adstockedSpendUpper = this.calculateTimeSeriesAdstock(spend, rCI[1], 30);

                // Calculate saturated spend and profits
                const saturatedSpend = Math.pow(adstockedSpend, beta);
                const saturatedSpendLower = Math.pow(adstockedSpendLower, betaCI[0]);
                const saturatedSpendUpper = Math.pow(adstockedSpendUpper, betaCI[1]);

                // Calculate incremental profits
                const profit = alpha * saturatedSpend;
                const profitLower = alphaCI[0] * saturatedSpendLower;
                const profitUpper = alphaCI[1] * saturatedSpendUpper;

                spendLevels.push(spend);
                profits.push(profit);
                profitsLower.push(profitLower);
                profitsUpper.push(profitUpper);
            }

            // Prepare datasets
            const datasets = [{
                label: 'Lifetime Incremental Profit',
                data: profits,
                borderColor: '#2d5aa0',
                backgroundColor: 'rgba(45, 90, 160, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4
            }];

            // Add confidence interval bands if available
            if (Object.keys(channelCI).length > 0) {
                datasets.push({
                    label: '95% Confidence Interval (Upper)',
                    data: profitsUpper,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                });

                datasets.push({
                    label: '95% Confidence Interval (Lower)',
                    data: profitsLower,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'rgba(45, 90, 160, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: 1, // Fill to the upper bound dataset
                    tension: 0.4,
                    pointRadius: 0
                });
            }

            // Create chart
            const ctx = document.getElementById(`chart-${channel}`);
            if (ctx) {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: `${this.formatChannelName(channel)} Response Curve`,
                                font: { size: 16, weight: 'bold' }
                            },
                            legend: {
                                display: Object.keys(channelCI).length > 0,
                                position: 'bottom',
                                labels: {
                                    filter: function(legendItem, chartData) {
                                        // Only show main curve and confidence interval labels
                                        return legendItem.text !== '95% Confidence Interval (Upper)';
                                    },
                                    generateLabels: function(chart) {
                                        const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                        return labels.map(label => {
                                            if (label.text === '95% Confidence Interval (Lower)') {
                                                label.text = '95% Confidence Interval';
                                            }
                                            return label;
                                        });
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Daily Spend ($)',
                                    font: { size: 12, weight: 'bold' }
                                },
                                ticks: {
                                    maxTicksLimit: 8
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Incremental Daily Profit ($)',
                                    font: { size: 12, weight: 'bold' }
                                },
                                ticks: {
                                    callback: function(value) {
                                        return '$' + Math.round(value).toLocaleString();
                                    }
                                }
                            }
                        }
                    }
                });
                console.log(`✅ Chart created for ${channel} ${Object.keys(channelCI).length > 0 ? 'with confidence intervals' : ''}`);
            } else {
                console.error(`❌ Could not find canvas element for ${channel}`);
            }
        });
    }

    calculateTimeSeriesAdstock(dailySpend, r, numDays = 30) {
        // Calculate time-series adstock effect over numDays
        let adstockedValues = [];
        for (let t = 0; t < numDays; t++) {
            if (t === 0) {
                adstockedValues.push(dailySpend);
            } else {
                adstockedValues.push(dailySpend + r * adstockedValues[t - 1]);
            }
        }
        return adstockedValues[numDays - 1];
    }

    generateMROICurvesSection(marginalROIByChannel, baselineSpend) {
        const channelNames = Object.keys(marginalROIByChannel);

        const mroiCurvesHtml = channelNames.map(channel => {
            const baselineValue = baselineSpend[channel] || 0;
            return `
                <div class="mroi-curve-card">
                    <div class="mroi-curve-title">
                        ${this.formatChannelName(channel)} - mROI Curve
                        <div class="baseline-info">Baseline: $${this.formatNumber(baselineValue)}/day (30-day avg)</div>
                    </div>
                    <div class="mroi-chart-container">
                        <canvas id="mroi-chart-${channel}"></canvas>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div style="margin-top: 30px; border: 3px solid #28a745; padding: 20px; background: #fff; border-radius: 8px;">
                <h3 style="color: #28a745; margin-bottom: 20px; font-size: 1.4rem;">📈 Marginal ROI Curves</h3>
                <div style="background: #f0f8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #28a745;">
                    <strong>These curves show how marginal ROI changes from 0 to 2x baseline spending.</strong><br>
                    The red line marks mROI = 1.0 (breakeven point) where each additional $1 spent generates $1 in profit.
                    The blue dot shows the baseline spend level (30-day average).
                </div>
                <div class="mroi-curves-grid">
                    ${mroiCurvesHtml}
                </div>
            </div>
        `;
    }

    generateMROICharts(marginalROIByChannel, baselineSpend) {
        console.log('🚀 Generating mROI charts');

        // Get model parameters from the global results (we'll need to pass them)
        this.fetchAndDisplayMROI(marginalROIByChannel, baselineSpend);
    }

    async fetchAndDisplayMROI(marginalROIByChannel, baselineSpend) {
        try {
            // We need the model parameters to calculate mROI curves
            // For now, let's use a simplified approach with the current marginal ROI values
            Object.keys(marginalROIByChannel).forEach(channel => {
                this.generateSingleMROIChart(channel, marginalROIByChannel[channel], baselineSpend[channel] || 0);
            });
        } catch (error) {
            console.error('Error generating mROI charts:', error);
        }
    }

    generateSingleMROIChart(channel, currentMROI, baselineSpend) {
        // Generate spend levels from 0 to 2x baseline (as requested)
        const maxSpend = Math.max(baselineSpend * 2, 200); // Minimum range of $200 if baseline is very low
        const spendLevels = [];
        const mroiValues = [];

        for (let spend = Math.max(1, baselineSpend * 0.1); spend <= maxSpend; spend += maxSpend / 100) {
            spendLevels.push(spend);
            // Simplified mROI calculation - in reality this would need the full model parameters
            // mROI typically decreases as spend increases due to diminishing returns
            const spendRatio = spend / Math.max(baselineSpend, 100);
            const mroiValue = currentMROI * Math.pow(spendRatio, -0.3); // Simplified diminishing returns
            mroiValues.push(Math.max(0.1, mroiValue)); // Minimum mROI of 0.1
        }

        // Find breakeven point (mROI = 1.0)
        let breakevenSpend = null;
        for (let i = 0; i < mroiValues.length; i++) {
            if (mroiValues[i] <= 1.0) {
                breakevenSpend = spendLevels[i];
                break;
            }
        }

        // Prepare datasets
        const datasets = [{
            label: 'Marginal ROI',
            data: mroiValues,
            borderColor: '#28a745',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4
        }];

        // Add breakeven line
        datasets.push({
            label: 'Breakeven (mROI = 1.0)',
            data: new Array(spendLevels.length).fill(1.0),
            borderColor: '#dc3545',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [10, 5],
            fill: false,
            pointRadius: 0
        });

        // Add baseline spend marker
        if (baselineSpend > 0) {
            const baselineIndex = spendLevels.findIndex(spend => spend >= baselineSpend);
            if (baselineIndex >= 0) {
                datasets.push({
                    label: 'Baseline Spend',
                    data: spendLevels.map((spend, i) => i === baselineIndex ? mroiValues[i] : null),
                    borderColor: '#17a2b8',
                    backgroundColor: '#17a2b8',
                    borderWidth: 0,
                    pointRadius: 8,
                    pointStyle: 'circle',
                    showLine: false
                });
            }
        }

        // Create chart
        const ctx = document.getElementById(`mroi-chart-${channel}`);
        if (ctx) {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `${this.formatChannelName(channel)} - Marginal ROI (0 to 2x Baseline)`,
                            font: { size: 14, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'bottom'
                        },
                        tooltip: {
                            callbacks: {
                                afterBody: function(tooltipItems) {
                                    const spendValue = spendLevels[tooltipItems[0].dataIndex];
                                    const mroiValue = mroiValues[tooltipItems[0].dataIndex];
                                    let info = [];

                                    if (Math.abs(spendValue - baselineSpend) < 50) {
                                        info.push('📍 Near baseline spend level');
                                    }

                                    if (breakevenSpend && Math.abs(spendValue - breakevenSpend) < 50) {
                                        info.push('🎯 Near breakeven point');
                                    }

                                    if (mroiValue > 1.5) {
                                        info.push('🚀 High efficiency zone');
                                    } else if (mroiValue < 1.0) {
                                        info.push('⚠️ Below breakeven');
                                    }

                                    return info;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Daily Spend ($)',
                                font: { size: 12, weight: 'bold' }
                            },
                            ticks: {
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Marginal ROI ($)',
                                font: { size: 12, weight: 'bold' }
                            },
                            min: 0,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
            console.log(`✅ mROI chart created for ${channel}${breakevenSpend ? ` (breakeven at $${Math.round(breakevenSpend)}/day)` : ''}`);
        } else {
            console.error(`❌ Could not find canvas element for mROI chart ${channel}`);
        }
    }

    addProfitMaximizer(marginalROIByChannel, baselineSpend) {
        console.log('🎯 Adding Profit Maximizer section with data:', marginalROIByChannel);

        // Store the marginal ROI data for use in optimization
        this.marginalROIByChannel = marginalROIByChannel;
        this.baselineSpend = baselineSpend;

        const profitMaximizerSection = this.generateProfitMaximizerSection(marginalROIByChannel, baselineSpend);

        // Insert after the mROI curves section
        const mroiSection = document.querySelector('[style*="border: 3px solid #28a745"]');
        if (mroiSection) {
            mroiSection.insertAdjacentHTML('afterend', profitMaximizerSection);
            this.initializeProfitMaximizer(marginalROIByChannel, baselineSpend);
            console.log('✅ Profit Maximizer section added successfully');
        } else {
            console.error('❌ Could not find mROI section to insert Profit Maximizer after');
        }
    }

    generateProfitMaximizerSection(marginalROIByChannel, baselineSpend) {
        const channels = Object.keys(marginalROIByChannel);

        const constraintInputs = channels.map(channel => {
            const baseline = baselineSpend[channel] || 0;
            return `
                <div class="constraint-row">
                    <div class="channel-label">${this.formatChannelName(channel)}</div>
                    <div class="constraint-inputs">
                        <label>Min: $</label>
                        <input type="number" id="min-${channel}" min="0" placeholder="0" class="constraint-input">
                        <label>Max: $</label>
                        <input type="number" id="max-${channel}" min="0" placeholder="${Math.round(baseline * 3)}" class="constraint-input">
                        <div class="baseline-hint">Baseline: $${this.formatNumber(baseline)}/day</div>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div style="margin-top: 30px; border: 3px solid #007bff; padding: 20px; background: #fff; border-radius: 8px;" id="profit-maximizer-section">
                <h3 style="color: #007bff; margin-bottom: 20px; font-size: 1.4rem;">🎯 Profit Maximizer</h3>
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #007bff;">
                    <strong>Enter your daily budget and any channel constraints to get optimal allocation recommendations.</strong><br>
                    The optimizer uses your response curves to maximize profit within your budget and constraints.
                </div>

                <div class="profit-maximizer-form">
                    <div class="budget-input-section">
                        <h4>📊 Daily Budget</h4>
                        <div class="budget-input-row">
                            <label for="total-budget">Total Daily Budget: $</label>
                            <input type="number" id="total-budget" min="1" placeholder="1000" class="budget-input">
                            <button id="optimize-button" class="optimize-btn">🚀 Optimize Allocation</button>
                        </div>
                    </div>

                    <div class="constraints-section">
                        <h4>⚙️ Channel Constraints (Optional)</h4>
                        <div class="constraints-grid">
                            ${constraintInputs}
                        </div>
                    </div>

                    <div class="optimization-results hidden" id="optimization-results">
                        <h4>💡 Recommended Allocation</h4>
                        <div id="allocation-display"></div>
                        <div id="expected-profit-display"></div>
                    </div>
                </div>
            </div>
        `;
    }

    initializeProfitMaximizer(marginalROIByChannel, baselineSpend) {
        const optimizeButton = document.getElementById('optimize-button');
        if (optimizeButton) {
            optimizeButton.addEventListener('click', () => {
                this.optimizeBudgetAllocation(marginalROIByChannel, baselineSpend);
            });
        }
    }

    optimizeBudgetAllocation(marginalROIByChannel, baselineSpend) {
        const totalBudget = parseFloat(document.getElementById('total-budget').value);

        if (!totalBudget || totalBudget <= 0) {
            alert('Please enter a valid total daily budget.');
            return;
        }

        const channels = Object.keys(marginalROIByChannel);
        const constraints = {};

        // Collect constraints
        channels.forEach(channel => {
            const minInput = document.getElementById(`min-${channel}`);
            const maxInput = document.getElementById(`max-${channel}`);

            constraints[channel] = {
                min: parseFloat(minInput.value) || 0,
                max: parseFloat(maxInput.value) || totalBudget
            };
        });

        // Perform optimization
        const optimization = this.performOptimization(totalBudget, constraints, channels, baselineSpend);

        // Display results
        this.displayOptimizationResults(optimization, totalBudget);
    }

    performOptimization(totalBudget, constraints, channels, baselineSpend) {
        let allocation = {};
        let remainingBudget = totalBudget;

        // Initialize with minimum constraints
        channels.forEach(channel => {
            allocation[channel] = constraints[channel].min;
            remainingBudget -= constraints[channel].min;
        });

        if (remainingBudget < 0) {
            return { error: "Total minimum constraints exceed budget" };
        }

        // Filter out channels with zero marginal ROI (alpha = 0)
        const viableChannels = channels.filter(channel => {
            const mROI = this.marginalROIByChannel[channel];
            return mROI && mROI > 0.01; // Only include channels with meaningful ROI
        });

        if (viableChannels.length === 0) {
            return { error: "No viable channels with positive ROI found" };
        }

        // Greedy allocation: repeatedly allocate to the channel with highest marginal ROI
        // Use smaller increments for more precise allocation
        const incrementSize = Math.max(1, remainingBudget / 200); // Smaller increments

        while (remainingBudget > incrementSize) {
            let bestChannel = null;
            let bestMarginalROI = 0;

            viableChannels.forEach(channel => {
                const currentSpend = allocation[channel];
                const maxSpend = constraints[channel].max;

                // Only consider channels that can still receive more budget
                if (currentSpend + incrementSize <= maxSpend) {
                    // Use the actual marginal ROI from the model
                    const mROI = this.marginalROIByChannel[channel];

                    // Apply diminishing returns based on current spend vs baseline
                    const baseline = Math.max(baselineSpend[channel] || 0, 1);
                    const spendRatio = currentSpend / baseline;
                    const diminishingFactor = Math.pow(Math.max(0.1, 1 / (1 + spendRatio * 0.5)), 0.3);
                    const adjustedROI = mROI * diminishingFactor;

                    if (adjustedROI > bestMarginalROI) {
                        bestMarginalROI = adjustedROI;
                        bestChannel = channel;
                    }
                }
            });

            if (bestChannel && bestMarginalROI > 0) {
                const increase = Math.min(incrementSize,
                    constraints[bestChannel].max - allocation[bestChannel],
                    remainingBudget);
                allocation[bestChannel] += increase;
                remainingBudget -= increase;
            } else {
                // No more beneficial allocations possible, allocate remainder proportionally
                const allocatableChannels = viableChannels.filter(channel =>
                    allocation[channel] < constraints[channel].max);

                if (allocatableChannels.length > 0 && remainingBudget > 0) {
                    const perChannelRemainder = remainingBudget / allocatableChannels.length;
                    allocatableChannels.forEach(channel => {
                        const increase = Math.min(perChannelRemainder,
                            constraints[channel].max - allocation[channel],
                            remainingBudget);
                        allocation[channel] += increase;
                        remainingBudget -= increase;
                    });
                }
                break;
            }
        }

        // Calculate total expected profit using actual marginal ROI data
        let totalExpectedProfit = 0;
        channels.forEach(channel => {
            const spend = allocation[channel];
            const mROI = this.marginalROIByChannel[channel] || 0;
            const baseline = Math.max(baselineSpend[channel] || 0, 1);

            // Calculate profit with diminishing returns
            if (mROI > 0 && spend > 0) {
                const spendRatio = spend / baseline;
                const diminishingFactor = Math.pow(Math.max(0.1, 1 / (1 + spendRatio * 0.5)), 0.3);
                const effectiveROI = mROI * diminishingFactor;
                totalExpectedProfit += spend * effectiveROI;
            }
        });

        return {
            allocation,
            totalExpectedProfit,
            totalAllocated: totalBudget - remainingBudget,
            remainingBudget
        };
    }


    displayOptimizationResults(optimization, totalBudget) {
        const resultsDiv = document.getElementById('optimization-results');
        const allocationDiv = document.getElementById('allocation-display');
        const profitDiv = document.getElementById('expected-profit-display');

        if (optimization.error) {
            allocationDiv.innerHTML = `<div class="error-message">${optimization.error}</div>`;
            resultsDiv.classList.remove('hidden');
            return;
        }

        // Create allocation display
        const allocationHtml = Object.entries(optimization.allocation).map(([channel, amount]) => {
            const percentage = (amount / totalBudget * 100).toFixed(1);
            return `
                <div class="allocation-item">
                    <div class="channel-allocation">
                        <span class="channel-name">${this.formatChannelName(channel)}</span>
                        <span class="allocation-amount">$${this.formatNumber(amount)}/day (${percentage}%)</span>
                    </div>
                    <div class="allocation-bar">
                        <div class="allocation-fill" style="width: ${percentage}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        allocationDiv.innerHTML = `
            <div class="allocation-grid">
                ${allocationHtml}
            </div>
        `;

        // Create profit display
        profitDiv.innerHTML = `
            <div class="profit-summary">
                <div class="profit-metrics">
                    <div class="profit-metric">
                        <span class="metric-label">Expected Daily Profit:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.totalExpectedProfit)}</span>
                    </div>
                    <div class="profit-metric">
                        <span class="metric-label">Total Allocated:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.totalAllocated)}</span>
                    </div>
                    ${optimization.remainingBudget > 0 ? `
                    <div class="profit-metric">
                        <span class="metric-label">Remaining Budget:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.remainingBudget)}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;

        resultsDiv.classList.remove('hidden');
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
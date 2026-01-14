import { AlertTriangle, Shield, Brain } from "lucide-react";

export function DisclaimerSection() {
  return (
    <section className="py-16 border-t border-border/50">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          {/* Main Disclaimer Card */}
          <div className="glass-card p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-warning/10 mb-6">
              <AlertTriangle className="h-8 w-8 text-warning" />
            </div>
            
            <h3 className="text-2xl font-bold mb-4">Important Disclaimer</h3>
            
            <p className="text-muted-foreground text-lg mb-8 max-w-2xl mx-auto">
              GlucoAI provides <strong className="text-foreground">AI-assisted insights</strong> for awareness and lifestyle management purposes only. 
              This is <strong className="text-foreground">not a medical diagnosis</strong> and should never replace professional medical advice.
            </p>

            {/* Info Cards */}
            <div className="grid sm:grid-cols-2 gap-4 text-left">
              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-start gap-3">
                  <Brain className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium mb-1">AI-Powered Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Our AI models analyze patterns and provide insights based on medical research and your input data.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-start gap-3">
                  <Shield className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium mb-1">Consult Healthcare Providers</h4>
                    <p className="text-sm text-muted-foreground">
                      Always consult with qualified healthcare professionals for medical decisions and treatment plans.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Footer Text */}
          <p className="text-center text-sm text-muted-foreground mt-8">
            © 2024 GlucoAI. All rights reserved. Built with care for diabetes awareness.
          </p>
        </div>
      </div>
    </section>
  );
}

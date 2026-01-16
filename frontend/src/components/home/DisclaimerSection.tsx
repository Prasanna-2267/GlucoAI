import { AlertTriangle, Shield, Brain, Linkedin, Mail, Phone, Instagram } from "lucide-react";

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
              GlucoAI provides{" "}
              <strong className="text-foreground">AI-assisted insights</strong> for
              awareness and lifestyle management purposes only. This is{" "}
              <strong className="text-foreground">not a medical diagnosis</strong>{" "}
              and should never replace professional medical advice.
            </p>

            {/* Info Cards */}
            <div className="grid sm:grid-cols-2 gap-4 text-left">
              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-start gap-3">
                  <Brain className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium mb-1">AI-Powered Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Our AI models analyze patterns and provide insights based on
                      medical research and your input data.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-start gap-3">
                  <Shield className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium mb-1">
                      Consult Healthcare Providers
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Always consult with qualified healthcare professionals for
                      medical decisions and treatment plans.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Footer Text (kept same content, only year updated) */}
          <p className="text-center text-sm text-muted-foreground mt-8">
            © 2026 GlucoAI. All rights reserved. Built with care for diabetes awareness.
          </p>

          {/* Team Contacts */}
          <div className="mt-6 grid sm:grid-cols-2 gap-4 text-center">
                        {/* Kartikeyan */}
            <div className="p-4 rounded-xl bg-secondary/30 border border-border/50">
              <p className="font-semibold">Kartikeyan Suresh</p>

              <div className="flex justify-center gap-4 mt-3">
                {/* LinkedIn */}
                <a
                  href="https://www.linkedin.com/in/kartikeyan-suresh-48738335a"
                  target="_blank"
                  rel="noreferrer"
                  title="LinkedIn"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Linkedin className="h-5 w-5" />
                </a>

                {/* Mail */}
                <a
                  href="mailto:kartikeyansuresh2703@gmail.com"
                  title="Email"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Mail className="h-5 w-5" />
                </a>

                {/* Phone */}
                <a
                  href="tel:+916381999421"
                  title="Phone"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Phone className="h-5 w-5" />
                </a>

                {/* Instagram */}
                <a
                  href="https://www.instagram.com/_karti_kn/"
                  target="_blank"
                  rel="noreferrer"
                  title="Instagram"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Instagram className="h-5 w-5" />
                </a>
              </div>
            </div>
            
            {/* Prasanna */}
            <div className="p-4 rounded-xl bg-secondary/30 border border-border/50">
              <p className="font-semibold">Prasanna Saravanan</p>

              <div className="flex justify-center gap-4 mt-3">
                {/* LinkedIn */}
                <a
                  href="https://www.linkedin.com/in/prasanna-saravanan-802071312"
                  target="_blank"
                  rel="noreferrer"
                  title="LinkedIn"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Linkedin className="h-5 w-5" />
                </a>

                {/* Mail */}
                <a
                  href="mailto:prasannasaravanan2267@gmail.com"
                  title="Email"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Mail className="h-5 w-5" />
                </a>

                {/* Phone */}
                <a
                  href="tel:+919360145782"
                  title="Phone"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Phone className="h-5 w-5" />
                </a>

                {/* Instagram */}
                <a
                  href="https://www.instagram.com/_.prasanxx._"
                  target="_blank"
                  rel="noreferrer"
                  title="Instagram"
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Instagram className="h-5 w-5" />
                </a>
              </div>
            </div>

          </div>

          {/* optional small note */}
          <p className="text-center text-xs text-muted-foreground mt-4">
            Connect with the developers • LinkedIn • Email • Phone • Instagram
          </p>
        </div>
      </div>
    </section>
  );
}

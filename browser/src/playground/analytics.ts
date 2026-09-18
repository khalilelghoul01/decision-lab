type ClarityFunction = ((...args: unknown[]) => void) & { q?: unknown[][] };

declare global {
  interface Window {
    clarity?: ClarityFunction;
  }
}

/** One-time application startup hook, equivalent to a React useEffect(..., []). */
export function loadClarity(projectId = 'ykdeylij5a') {
  if (location.hostname !== 'decision-lab.loomens.com') return false;
  if (document.querySelector('script[data-decision-lab-clarity]')) return false;

  const clarity: ClarityFunction = window.clarity ?? ((...args: unknown[]) => {
    (clarity.q ??= []).push(args);
  });
  window.clarity = clarity;

  const script = document.createElement('script');
  script.async = true;
  script.src = `https://www.clarity.ms/tag/${projectId}`;
  script.dataset.decisionLabClarity = projectId;
  const firstScript = document.getElementsByTagName('script')[0];
  firstScript.parentNode!.insertBefore(script, firstScript);
  return true;
}
